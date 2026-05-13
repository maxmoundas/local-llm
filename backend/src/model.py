"""
Model loading and inference wrapper for Gemma 4 (and compatible HuggingFace models).

Quantization: 4-bit NF4 via bitsandbytes.  Vision layers (vision_tower,
multi_modal_projector) are kept in bfloat16 so image understanding works
correctly — see _build_bnb_config for the full explanation.
"""
from __future__ import annotations

import threading
import time
from collections.abc import Iterator
from typing import Any

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoProcessor,
    AutoTokenizer,
    BitsAndBytesConfig,
    TextIteratorStreamer,
)

from src.logger import get_logger

logger = get_logger(__name__)


class LocalLLM:
    def __init__(self, cfg: dict[str, Any]) -> None:
        self.cfg = cfg["model"]
        self.model: AutoModelForCausalLM | None = None
        self.processor: Any = None
        self._lock = threading.Lock()

    @property
    def tokenizer(self) -> Any:
        if self.processor is None:
            return None
        return getattr(self.processor, "tokenizer", self.processor)

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def load(self) -> None:
        repo_id = self.cfg["repo_id"]
        quantization = self.cfg.get("quantization", "4bit")
        cache_dir = self.cfg.get("cache_dir", "./models")

        logger.info("Loading processor from %s", repo_id)
        try:
            self.processor = AutoProcessor.from_pretrained(
                repo_id, cache_dir=cache_dir, trust_remote_code=False
            )
            logger.info("Loaded AutoProcessor")
        except Exception:
            self.processor = AutoTokenizer.from_pretrained(
                repo_id, cache_dir=cache_dir, trust_remote_code=False
            )
            logger.info("Loaded AutoTokenizer (fallback)")

        bnb_config = self._build_bnb_config(quantization)
        dtype = torch.bfloat16 if quantization == "none" else None

        # Use a single explicit GPU map when quantizing.  device_map="auto"
        # triggers accelerate CPU-offload hooks that are incompatible with
        # bitsandbytes Params4bit on this system.
        device_map = {"": "cuda:0"} if bnb_config is not None else self.cfg.get("device_map", "auto")

        logger.info("Loading model %s | quantization=%s", repo_id, quantization)
        t0 = time.perf_counter()
        self.model = AutoModelForCausalLM.from_pretrained(
            repo_id,
            cache_dir=cache_dir,
            quantization_config=bnb_config,
            device_map=device_map,
            dtype=dtype,
            trust_remote_code=False,
            low_cpu_mem_usage=True,
        )
        self.model.eval()
        logger.info("Model loaded in %.1fs | class=%s", time.perf_counter() - t0, type(self.model).__name__)

    def _build_bnb_config(self, quantization: str) -> BitsAndBytesConfig | None:
        if quantization == "4bit":
            # should_convert_module() uses re.match() anchored at the start of
            # the full module path (e.g. "model.vision_tower.encoder.layers...").
            # Short names like "vision_tower" never match — full path prefixes
            # are required.  Keeping the vision encoder and projector in bfloat16
            # prevents 4-bit from collapsing all images to identical embeddings.
            skip = ["model.vision_tower", "model.multi_modal_projector", "lm_head"]
            return BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                llm_int8_skip_modules=skip,
            )
        if quantization == "8bit":
            skip = ["model.vision_tower", "model.multi_modal_projector", "lm_head"]
            return BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_skip_modules=skip,
            )
        return None

    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------

    def _to_device(self, inputs: dict) -> dict:
        device = next(self.model.parameters()).device
        return {
            k: v.to(device=device, dtype=torch.bfloat16) if v.is_floating_point() else v.to(device)
            for k, v in inputs.items()
        }

    def _build_text_inputs(self, messages: list[dict]) -> dict[str, torch.Tensor]:
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        if hasattr(self.processor, "tokenizer"):
            inputs = self.processor(text=text, return_tensors="pt")
        else:
            inputs = self.processor(text, return_tensors="pt")
        device = next(self.model.parameters()).device
        return {k: v.to(device) for k, v in inputs.items() if k in ("input_ids", "attention_mask")}

    def _build_vision_inputs(self, messages: list[dict]) -> dict:
        """Build inputs for a messages list that already contains image content dicts."""
        inputs = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )
        pv = inputs.get("pixel_values")
        logger.info(
            "Vision inputs | input_ids=%s | pixel_values=%s | pv_sum=%.0f",
            inputs["input_ids"].shape,
            pv.shape if pv is not None else "MISSING",
            float(pv.float().sum()) if pv is not None else 0.0,
        )
        return self._to_device(inputs)

    # ------------------------------------------------------------------
    # Text inference
    # ------------------------------------------------------------------

    def generate(
        self,
        messages: list[dict],
        max_new_tokens: int | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        repetition_penalty: float | None = None,
    ) -> str:
        assert self.model is not None and self.processor is not None

        max_new_tokens = max_new_tokens or self.cfg.get("max_new_tokens", 512)
        temperature = temperature if temperature is not None else self.cfg.get("temperature", 0.7)
        top_p = top_p if top_p is not None else self.cfg.get("top_p", 0.9)
        repetition_penalty = repetition_penalty or self.cfg.get("repetition_penalty", 1.1)
        do_sample = self.cfg.get("do_sample", True) and temperature > 0

        inputs = self._build_text_inputs(messages)
        prompt_len = inputs["input_ids"].shape[-1]

        t0 = time.perf_counter()
        with self._lock, torch.inference_mode():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=temperature if do_sample else None,
                top_p=top_p if do_sample else None,
                repetition_penalty=repetition_penalty,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        new_ids = output_ids[0][prompt_len:]
        text = self.tokenizer.decode(new_ids, skip_special_tokens=True)
        elapsed = time.perf_counter() - t0
        logger.info("Generation done | tokens=%d | %.1fs | %.1f tok/s", len(new_ids), elapsed, len(new_ids) / elapsed)
        return text.strip()

    def stream(
        self,
        messages: list[dict],
        max_new_tokens: int | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        repetition_penalty: float | None = None,
    ) -> Iterator[str]:
        assert self.model is not None and self.processor is not None

        max_new_tokens = max_new_tokens or self.cfg.get("max_new_tokens", 512)
        temperature = temperature if temperature is not None else self.cfg.get("temperature", 0.7)
        top_p = top_p if top_p is not None else self.cfg.get("top_p", 0.9)
        repetition_penalty = repetition_penalty or self.cfg.get("repetition_penalty", 1.1)
        do_sample = self.cfg.get("do_sample", True) and temperature > 0

        inputs = self._build_text_inputs(messages)
        streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)

        gen_kwargs = dict(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature if do_sample else None,
            top_p=top_p if do_sample else None,
            repetition_penalty=repetition_penalty,
            pad_token_id=self.tokenizer.eos_token_id,
            streamer=streamer,
        )
        thread = threading.Thread(target=self._generate_in_thread, args=(gen_kwargs,))
        thread.start()
        for token in streamer:
            yield token
        thread.join()

    # ------------------------------------------------------------------
    # Vision inference
    # ------------------------------------------------------------------

    def _inject_image(self, messages: list[dict], image: "PIL.Image.Image") -> list[dict]:
        """Return a copy of messages with the PIL image inserted into the last user turn."""
        msgs = list(messages)
        for i in range(len(msgs) - 1, -1, -1):
            if msgs[i]["role"] == "user":
                msgs[i] = {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": msgs[i]["content"]},
                    ],
                }
                break
        return msgs

    def stream_with_image(
        self,
        image: "PIL.Image.Image",
        messages: list[dict],
        max_new_tokens: int | None = None,
    ) -> Iterator[str]:
        assert self.model is not None and self.processor is not None
        assert hasattr(self.processor, "tokenizer"), "stream_with_image requires AutoProcessor"

        max_new_tokens = max_new_tokens or self.cfg.get("max_new_tokens", 512)
        temperature = self.cfg.get("temperature", 0.7)
        top_p = self.cfg.get("top_p", 0.9)
        repetition_penalty = self.cfg.get("repetition_penalty", 1.1)
        do_sample = self.cfg.get("do_sample", True) and temperature > 0

        msgs = self._inject_image(messages, image)
        inputs = self._build_vision_inputs(msgs)

        streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)
        gen_kwargs = dict(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature if do_sample else None,
            top_p=top_p if do_sample else None,
            repetition_penalty=repetition_penalty,
            pad_token_id=self.tokenizer.eos_token_id,
            streamer=streamer,
        )
        thread = threading.Thread(target=self._generate_in_thread, args=(gen_kwargs,))
        thread.start()
        for token in streamer:
            yield token
        thread.join()

    def generate_with_image(
        self,
        image: "PIL.Image.Image",
        prompt: str,
        max_new_tokens: int = 512,
    ) -> str:
        assert self.model is not None and self.processor is not None
        assert hasattr(self.processor, "tokenizer"), "generate_with_image requires AutoProcessor"

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        inputs = self._build_vision_inputs(messages)
        prompt_len = inputs["input_ids"].shape[-1]

        t0 = time.perf_counter()
        with self._lock, torch.inference_mode():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        new_ids = output_ids[0][prompt_len:]
        response = self.tokenizer.decode(new_ids, skip_special_tokens=True).strip()
        logger.info("Vision done | tokens=%d | %.1fs", len(new_ids), time.perf_counter() - t0)
        return response

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def _generate_in_thread(self, gen_kwargs: dict) -> None:
        with self._lock, torch.inference_mode():
            self.model.generate(**gen_kwargs)

    def count_tokens(self, text: str) -> int:
        return len(self.tokenizer.encode(text))

    @property
    def is_loaded(self) -> bool:
        return self.model is not None and self.processor is not None

    @property
    def repo_id(self) -> str:
        return self.cfg["repo_id"]
