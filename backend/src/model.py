"""
Model loading and inference wrapper for HuggingFace models.

Supports 4-bit / 8-bit quantization via bitsandbytes and automatic
GPU+CPU memory splitting via accelerate's device_map="auto".

Uses AutoProcessor (handles multimodal models like Gemma 4) with a
text-only input path — pixel_values are never required for plain chat.
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
        self.processor: Any = None   # AutoProcessor or AutoTokenizer
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Convenience: always get a tokenizer-like object for encode/decode
    # ------------------------------------------------------------------

    @property
    def tokenizer(self) -> Any:
        if self.processor is None:
            return None
        # Processors wrap a tokenizer; plain tokenizers are their own tokenizer
        return getattr(self.processor, "tokenizer", self.processor)

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def load(self) -> None:
        repo_id = self.cfg["repo_id"]
        quantization = self.cfg.get("quantization", "4bit")
        device_map = self.cfg.get("device_map", "auto")
        cache_dir = self.cfg.get("cache_dir", "./models")

        logger.info("Loading processor from %s", repo_id)
        # Prefer AutoProcessor (required for multimodal models like Gemma 4);
        # fall back to AutoTokenizer for plain text-only models.
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

        logger.info(
            "Loading model %s | quantization=%s | device_map=%s",
            repo_id,
            quantization,
            device_map,
        )
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
        elapsed = time.perf_counter() - t0
        logger.info("Model loaded in %.1fs", elapsed)
        self._log_device_map()

    def _build_bnb_config(self, quantization: str) -> BitsAndBytesConfig | None:
        if quantization == "4bit":
            return BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            )
        if quantization == "8bit":
            return BitsAndBytesConfig(load_in_8bit=True)
        return None

    def _log_device_map(self) -> None:
        if not hasattr(self.model, "hf_device_map"):
            return
        device_counts: dict[str, int] = {}
        for device in self.model.hf_device_map.values():
            device_counts[str(device)] = device_counts.get(str(device), 0) + 1
        logger.info("Device map summary: %s", device_counts)

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def _build_input(self, messages: list[dict[str, str]]) -> dict[str, torch.Tensor]:
        tok = self.tokenizer
        text = tok.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        # Use processor for encoding (handles multimodal models transparently).
        # Pass text= kwarg so multimodal processors don't require pixel_values.
        if hasattr(self.processor, "tokenizer"):
            inputs = self.processor(text=text, return_tensors="pt")
        else:
            inputs = self.processor(text, return_tensors="pt")

        device = next(self.model.parameters()).device
        result = {"input_ids": inputs["input_ids"].to(device)}
        if "attention_mask" in inputs:
            result["attention_mask"] = inputs["attention_mask"].to(device)
        return result

    def generate(
        self,
        messages: list[dict[str, str]],
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

        inputs = self._build_input(messages)
        prompt_len = inputs["input_ids"].shape[-1]

        logger.debug(
            "Generating | prompt_tokens=%d | max_new=%d | temp=%.2f",
            prompt_len,
            max_new_tokens,
            temperature,
        )
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
        tps = len(new_ids) / elapsed if elapsed > 0 else 0

        logger.info(
            "Generation done | new_tokens=%d | elapsed=%.2fs | %.1f tok/s",
            len(new_ids),
            elapsed,
            tps,
        )
        return text.strip()

    def stream(
        self,
        messages: list[dict[str, str]],
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

        inputs = self._build_input(messages)
        streamer = TextIteratorStreamer(
            self.tokenizer, skip_prompt=True, skip_special_tokens=True
        )

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

    def _generate_in_thread(self, gen_kwargs: dict) -> None:
        with self._lock, torch.inference_mode():
            self.model.generate(**gen_kwargs)

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def count_tokens(self, text: str) -> int:
        return len(self.tokenizer.encode(text))

    @property
    def is_loaded(self) -> bool:
        return self.model is not None and self.processor is not None

    @property
    def repo_id(self) -> str:
        return self.cfg["repo_id"]
