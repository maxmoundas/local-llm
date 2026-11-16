#!/usr/bin/env python3
"""
Minimal Gemma 4 vision probe — same loading path as text inference,
with an image added to the message.

Usage (from the backend/ directory):
    python scripts/probe_vision.py
    python scripts/probe_vision.py tests/images/dog.png
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor, BitsAndBytesConfig

REPO_ID = "google/gemma-4-E4B-it"
CACHE_DIR = "../models"
PROMPT = "Describe what you see in this image."
MAX_NEW_TOKENS = 200

# Full module path prefixes — re.match() anchors at string start, so
# "vision_tower" alone never matches "model.vision_tower.encoder...".
# Using the full prefix guarantees the skip is respected.
SKIP_MODULES = ["model.vision_tower", "model.multi_modal_projector", "lm_head"]


def load_model():
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        llm_int8_skip_modules=SKIP_MODULES,
    )

    print(f"Loading processor ...")
    processor = AutoProcessor.from_pretrained(REPO_ID, cache_dir=CACHE_DIR)

    print(f"Loading model (4-bit, skip={SKIP_MODULES}) ...")
    t0 = time.perf_counter()
    model = AutoModelForCausalLM.from_pretrained(
        REPO_ID,
        cache_dir=CACHE_DIR,
        quantization_config=bnb,
        # Force all layers onto the single GPU. device_map="auto" triggers
        # accelerate CPU-offload hooks that are incompatible with the installed
        # bitsandbytes version (Params4bit.__new__ rejects _is_hf_initialized).
        device_map={"": "cuda:0"},
        low_cpu_mem_usage=True,
    )
    model.eval()
    print(f"Model loaded in {time.perf_counter() - t0:.1f}s")
    return model, processor


def check_vision_layers(model) -> None:
    quant, clean = [], []
    for name, module in model.named_modules():
        cls = type(module).__name__
        if "vision_tower" not in name and "multi_modal_projector" not in name:
            continue
        if not hasattr(module, "weight"):
            continue
        if "4bit" in cls.lower() or "8bit" in cls.lower():
            quant.append(f"  QUANTIZED  {name}: {cls}")
        else:
            clean.append(f"  ok         {name}: {cls}")

    if quant:
        print(f"\n  *** {len(quant)} vision layer(s) are still quantized — descriptions will be wrong ***")
        for line in quant[:4]:
            print(line)
    else:
        print(f"\n  Vision layers not quantized ({len(clean)} layers in bfloat16) — correct")
    print()


def describe_image(model, processor, image_path: Path) -> None:
    image = Image.open(image_path).convert("RGB")
    print(f"  {image_path.name}  ({image.size[0]}x{image.size[1]} px)")

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": PROMPT},
            ],
        }
    ]

    # Single-step: apply_chat_template handles image token insertion and
    # returns pixel_values alongside input_ids.
    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    )

    pv = inputs.get("pixel_values")
    mm = inputs.get("mm_token_type_ids")
    print(f"  pixel_values: {pv.shape if pv is not None else 'MISSING'}  "
          f"pv_sum={float(pv.float().sum()):.0f}  "
          f"image_tokens={int(mm.sum()) if mm is not None else 'N/A'}")

    device = next(model.parameters()).device
    inputs = {
        k: v.to(device=device, dtype=torch.bfloat16) if v.is_floating_point() else v.to(device)
        for k, v in inputs.items()
    }

    t0 = time.perf_counter()
    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            pad_token_id=processor.tokenizer.eos_token_id,
        )

    prompt_len = inputs["input_ids"].shape[-1]
    new_ids = output_ids[0][prompt_len:]
    response = processor.tokenizer.decode(new_ids, skip_special_tokens=True).strip()
    print(f"  {time.perf_counter() - t0:.1f}s | {len(new_ids)} tokens\n")
    print(f"  Response: {response}\n")


def main() -> None:
    image_paths = (
        [Path(p) for p in sys.argv[1:]]
        if len(sys.argv) > 1
        else sorted(Path("tests/images").glob("*"))
    )
    image_paths = [p for p in image_paths if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp", ".bmp"}]

    if not image_paths:
        print("No images found. Add images to tests/images/ or pass paths as arguments.", file=sys.stderr)
        sys.exit(1)

    model, processor = load_model()
    check_vision_layers(model)

    for image_path in image_paths:
        print("=" * 60)
        describe_image(model, processor, image_path)

    print("Done.")


if __name__ == "__main__":
    main()
