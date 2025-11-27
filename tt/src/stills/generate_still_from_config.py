#!/usr/bin/env python
import argparse
import json
import logging
import re
from pathlib import Path
import sys
import time

import torch
from diffusers import StableDiffusionPipeline
from transformers import CLIPTokenizer


# --- logging setup -----------------------------------------------------------

SCRIPT_PATH = Path(__file__).resolve()
TT_ROOT = SCRIPT_PATH.parents[2]  # .../tt
LOG_DIR = TT_ROOT / "assets" / "stills"
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_PATH = LOG_DIR / "generate_still_debug.log"


def setup_logger() -> logging.Logger:
    logger = logging.getLogger("generate_still")
    logger.setLevel(logging.DEBUG)

    if not logger.handlers:
        fmt = "[%(asctime)s] %(levelname)s: %(message)s"
        formatter = logging.Formatter(fmt, datefmt="%Y-%m-%d %H:%M:%S")

        fh = logging.FileHandler(LOG_PATH, mode="a", encoding="utf-8")
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(logging.INFO)
        ch.setFormatter(formatter)
        logger.addHandler(ch)

    return logger


logger = setup_logger()
logger.debug(f"SCRIPT_PATH={SCRIPT_PATH}")
logger.debug(f"TT_ROOT={TT_ROOT}")
logger.debug(f"LOG_PATH={LOG_PATH}")


# --- core functions ----------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate still images with AbsoluteReality using a Yazzie-style JSON config."
    )
    parser.add_argument(
        "config",
        help="Path to JSON file with prompt and generation settings.",
    )
    return parser.parse_args()


def load_config(path: Path) -> dict:
    logger.info(f"Loading config from: {path}")
    if not path.is_file():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        cfg = json.load(f)

    # Support BOTH legacy Yazzie fields and newer ones
    # char_name -> character_name
    if "character_name" not in cfg and "char_name" in cfg:
        cfg["character_name"] = cfg["char_name"]
        logger.info(f"Using char_name='{cfg['char_name']}' as character_name.")

    # seed_start optional; default to 0 for deterministic but simple behavior
    if "seed_start" not in cfg:
        cfg["seed_start"] = 0
        logger.info("seed_start not found in config; defaulting to 0.")

    # Some configs may not have negative_prompt; default to empty string
    if "negative_prompt" not in cfg:
        cfg["negative_prompt"] = ""
        logger.info("negative_prompt not found in config; defaulting to empty string.")

    required_keys = [
        "character_name",
        "prompt",
        "num_images",
        "steps",
        "guidance_scale",
        "width",
        "height",
        "seed_start",
    ]
    missing = [k for k in required_keys if k not in cfg]
    if missing:
        # This is now a *user error*, not a hard crash.
        raise ValueError(f"Missing required key(s) in config: {', '.join(missing)}")

    logger.debug(f"Config contents: {cfg}")
    return cfg


def get_next_index(output_dir: Path) -> int:
    """
    Scan existing files like 000.png, 001.png, etc. and return the next index.
    If none exist, returns 0.
    """
    logger.info(f"Determining next index in output dir: {output_dir}")
    pattern = re.compile(r"^(\d{3})\.png$")
    max_idx = -1

    if not output_dir.exists():
        logger.debug("Output directory does not exist yet; starting at 0.")
        return 0

    for p in output_dir.iterdir():
        if p.is_file():
            match = pattern.match(p.name)
            if match:
                idx = int(match.group(1))
                max_idx = max(max_idx, idx)

    next_idx = max_idx + 1
    logger.info(f"Next index: {next_idx:03d}")
    return next_idx


def main() -> int:
    logger.info("=== generate_still_from_config: START ===")

    args = parse_args()
    logger.info(f"Raw args: {args}")

    config_path = Path(args.config).expanduser().resolve()
    logger.info(f"Resolved config path: {config_path}")

    try:
        cfg = load_config(config_path)
    except Exception as e:
        logger.error(f"Error loading config: {e}")
        logger.info("Fix the JSON file and re-run; environment remains usable.")
        # Do NOT re-raise; just exit gracefully
        return 0

    # Pull fields from config (after normalization)
    character_name: str = cfg["character_name"]
    prompt: str = cfg["prompt"]
    negative_prompt: str = cfg.get("negative_prompt", "")
    num_images: int = int(cfg["num_images"])
    steps: int = int(cfg["steps"])
    guidance_scale: float = float(cfg["guidance_scale"])
    width: int = int(cfg["width"])
    height: int = int(cfg["height"])
    seed_start: int = int(cfg["seed_start"])

    logger.info(f"Character: {character_name}")
    logger.info(f"Prompt: {prompt}")
    logger.info(f"Negative prompt: {negative_prompt}")
    logger.info(
        f"num_images={num_images}, steps={steps}, "
        f"guidance_scale={guidance_scale}, width={width}, height={height}, "
        f"seed_start={seed_start}"
    )

    # Resolve tt root (current file: .../tt/src/stills/)
    tt_root = TT_ROOT
    assets_root = tt_root / "assets"
    stills_root = assets_root / "stills" / character_name
    stills_root.mkdir(parents=True, exist_ok=True)
    logger.info(f"tt_root={tt_root}")
    logger.info(f"Assets root={assets_root}")
    logger.info(f"Output directory (stills_root)={stills_root}")

    # Model path from env setup
    model_path = Path.home() / "absreality_pipeline" / "models" / "AbsoluteReality"
    logger.info(f"Model path: {model_path}")
    if not model_path.exists():
        logger.error(f"Model directory not found at {model_path}")
        logger.info("Check your setup script / model download and re-run.")
        return 0

    # Device / dtype
    has_cuda = torch.cuda.is_available()
    device = "cuda" if has_cuda else "cpu"
    dtype = torch.float16 if has_cuda else torch.float32
    logger.info(f"Using device={device}, dtype={dtype}, cuda_available={has_cuda}")

    logger.info("Loading AbsoluteReality pipeline...")
    try:
        pipe = StableDiffusionPipeline.from_pretrained(
            model_path,
            torch_dtype=dtype,
            safety_checker=None,
        ).to(device)
    except Exception as e:
        logger.error(f"Failed to load pipeline: {e}")
        logger.info("Fix the environment/model and re-run.")
        return 0

    logger.info("Pipeline loaded successfully.")

    # Validate prompt length against CLIP's 77 token limit
    logger.info("Validating prompt token length...")
    tokenizer = pipe.tokenizer
    prompt_tokens = tokenizer(prompt, return_tensors="pt")
    prompt_length = prompt_tokens.input_ids.shape[1]
    
    if prompt_length > 77:
        logger.error(
            f"Prompt exceeds CLIP's maximum token length!\n"
            f"  Current length: {prompt_length} tokens\n"
            f"  Maximum allowed: 77 tokens\n"
            f"  Excess: {prompt_length - 77} tokens\n"
            f"  Please shorten your prompt and try again."
        )
        return 1
    
    logger.info(f"Prompt token length: {prompt_length}/77 ✓")
    
    # Also check negative prompt if present
    if negative_prompt:
        neg_tokens = tokenizer(negative_prompt, return_tensors="pt")
        neg_length = neg_tokens.input_ids.shape[1]
        
        if neg_length > 77:
            logger.error(
                f"Negative prompt exceeds CLIP's maximum token length!\n"
                f"  Current length: {neg_length} tokens\n"
                f"  Maximum allowed: 77 tokens\n"
                f"  Excess: {neg_length - 77} tokens\n"
                f"  Please shorten your negative prompt and try again."
            )
            return 1
        
        logger.info(f"Negative prompt token length: {neg_length}/77 ✓")

    # Determine starting index for output filenames
    start_index = get_next_index(stills_root)

    for i in range(num_images):
        idx = start_index + i
        filename = f"{idx:03d}.png"
        out_path = stills_root / filename

        seed = seed_start + i
        generator = torch.Generator(device=device).manual_seed(seed)

        logger.info(
            f"Generating {character_name} frame {idx:03d} "
            f"(seed={seed}) → {out_path}"
        )

        start_time = time.time()
        result = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt or None,
            num_inference_steps=steps,
            guidance_scale=guidance_scale,
            width=width,
            height=height,
            generator=generator,
        )
        elapsed_time = time.time() - start_time
        minutes = int(elapsed_time // 60)
        seconds = elapsed_time % 60

        image = result.images[0]
        image.save(out_path)
        logger.info(f"Saved {out_path} (generation time: {minutes}m {seconds:.2f}s)")

    logger.info(
        f"=== generate_still_from_config: DONE, generated {num_images} image(s) "
        f"into {stills_root} ==="
    )
    return 0


if __name__ == "__main__":
    # main() handles its own errors and logs them; we don't re-raise.
    sys.exit(main())
