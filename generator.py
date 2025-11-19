# generator.py
"""
Generator utilities for AI Image Generator app.

Improvements in this version:
- Consistent run folder layout: outputs/YYYYMMDD/HH-MM-SS
- Central batch metadata (metadata.json)
- Per-image metadata JSON retained
- Per-step callback for progress reporting (when supported)
- GLOBAL CSV log at outputs/logs/generations.csv (one row per generated image)
"""

import os
import json
import csv
import uuid
import time
import inspect
import warnings
from datetime import datetime
from typing import Callable, Optional, Tuple, List, Dict

from PIL import Image, ImageDraw, ImageFont
import torch
from diffusers import StableDiffusionPipeline

# Canonical output folder
OUTPUT_ROOT = "outputs"
LOGS_DIR = os.path.join(OUTPUT_ROOT, "logs")
GLOBAL_LOG_CSV = os.path.join(LOGS_DIR, "generations.csv")

os.makedirs(OUTPUT_ROOT, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)


def _make_run_folder() -> str:
    """
    Create a timestamped subfolder under OUTPUT_ROOT for one run.
    Uses local time (datetime.now) instead of UTC so folder date follows server locale.
    Format: outputs/YYYYMMDD/HH-MM-SS
    """
    now = datetime.now()
    date = now.strftime("%Y%m%d")
    timepart = now.strftime("%H-%M-%S")
    folder = os.path.join(OUTPUT_ROOT, date, timepart)
    os.makedirs(folder, exist_ok=True)
    return folder

def _append_to_global_csv(row: dict):
    """
    Append a row (dict) to the global CSV. Creates file with header if missing.
    Expected keys (will be written in this order):
    timestamp, run_id, index, filename, path, prompt, negative_prompt, style,
    model_id, seed, steps, guidance, elapsed
    """
    fieldnames = [
        "timestamp", "run_id", "index", "filename", "path",
        "prompt", "negative_prompt", "style", "model_id",
        "seed", "steps", "guidance", "elapsed"
    ]

    write_header = not os.path.exists(GLOBAL_LOG_CSV) or os.path.getsize(GLOBAL_LOG_CSV) == 0
    try:
        with open(GLOBAL_LOG_CSV, mode="a", newline="", encoding="utf-8") as csvf:
            writer = csv.DictWriter(csvf, fieldnames=fieldnames)
            if write_header:
                writer.writeheader()
            # make sure all fields are present (avoid KeyError)
            safe_row = {k: row.get(k, "") for k in fieldnames}
            writer.writerow(safe_row)
    except Exception:
        # don't fail generation process if logging fails
        pass


def load_pipeline(model_id: str = "runwayml/stable-diffusion-v1-5", device: Optional[str] = None) -> StableDiffusionPipeline:
    """
    Robust loader: chooses 'dtype' or 'torch_dtype' based on diffusers signature,
    moves pipeline to device, and applies memory/perf optimizations.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # prefer float16 on GPU
    dtype_val = torch.float16 if device == "cuda" else torch.float32

    # prepare safe kwargs (low memory usage)
    base_kwargs = {"low_cpu_mem_usage": True}

    # inspect from_pretrained signature to determine arg name
    fp = StableDiffusionPipeline.from_pretrained
    try:
        sig = inspect.signature(fp)
        params = sig.parameters
        if "dtype" in params:
            base_kwargs["dtype"] = dtype_val
        elif "torch_dtype" in params:
            base_kwargs["torch_dtype"] = dtype_val
        else:
            # fallback to torch_dtype for older versions
            base_kwargs["torch_dtype"] = dtype_val
    except Exception:
        base_kwargs["torch_dtype"] = dtype_val

    # attempt load; raise helpful error on failure
    try:
        # avoid showing noisy HF warnings here
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pipe = StableDiffusionPipeline.from_pretrained(model_id, **base_kwargs)
    except Exception as e:
        raise RuntimeError(f"Failed to load model '{model_id}': {e}") from e

    # move to device
    try:
        pipe = pipe.to(device)
    except Exception:
        pass

    # optimizations
    try:
        pipe.enable_attention_slicing()
    except Exception:
        pass

    # try enable xformers if available (best-effort)
    try:
        import xformers  # noqa: F401
        try:
            pipe.enable_xformers_memory_efficient_attention()
        except Exception:
            pass
    except Exception:
        pass

    # store model id for metadata
    try:
        pipe.model_id = model_id
    except Exception:
        pass

    return pipe


def add_watermark(img: Image.Image, text: str = "AI-Generated", opacity: int = 180, margin: int = 10) -> Image.Image:
    """
    Add a semi-transparent watermark text at bottom-left corner.

    Args:
        img: PIL Image
        text: watermark text
        opacity: 0-255 alpha for watermark text
        margin: pixels from left/bottom edges

    Returns:
        New PIL Image (RGB)
    """
    if img.mode != "RGBA":
        base = img.convert("RGBA")
    else:
        base = img.copy()

    watermark = Image.new("RGBA", base.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(watermark)
    try:
        font = ImageFont.truetype("arial.ttf", 18)
    except Exception:
        font = ImageFont.load_default()

    x = margin
    y = base.height - 20 - margin
    draw.text((x, y), text, fill=(255, 255, 255, opacity), font=font)

    combined = Image.alpha_composite(base, watermark)
    return combined.convert("RGB")


def save_image_with_meta(
    image: Image.Image,
    meta: dict,
    out_dir: Optional[str] = None,
    img_format: str = "PNG",
    jpeg_quality: int = 95,
    filename: Optional[str] = None
) -> Tuple[str, str, str]:
    """
    Save image and metadata JSON to disk inside out_dir.

    Returns:
        (filename, path, meta_path)
    """
    if out_dir is None:
        out_dir = _make_run_folder()
    else:
        os.makedirs(out_dir, exist_ok=True)

    timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    uid = str(uuid.uuid4())[:8]

    if filename is None:
        base = f"{timestamp}_{uid}"
    else:
        base = filename

    if img_format.upper() in ("JPEG", "JPG"):
        filename_full = f"{base}.jpg"
        path = os.path.join(out_dir, filename_full)
        image.save(path, format="JPEG", quality=jpeg_quality)
    else:
        filename_full = f"{base}.png"
        path = os.path.join(out_dir, filename_full)
        image.save(path, format="PNG")

    meta_path = path + ".json"
    meta_copy = dict(meta)
    meta_copy.update({"filename": filename_full, "path": path, "saved_at": datetime.utcnow().isoformat() + "Z"})

    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta_copy, f, ensure_ascii=False, indent=2)

    return filename_full, path, meta_path


def generate_images(
    pipe: StableDiffusionPipeline,
    prompt: str,
    style: Optional[str] = None,
    negative_prompt: Optional[str] = None,
    n_images: int = 1,
    guidance_scale: float = 7.5,
    num_inference_steps: int = 30,
    seed: Optional[int] = None,
    add_wm: bool = True,
    watermark_text: str = "AI-Generated",
    out_format: str = "PNG",
    progress_callback: Optional[Callable[[float, str], None]] = None,
    out_dir: Optional[str] = None
) -> List[Dict]:
    """
    Generate images with the provided pipeline.

    All images for this run are saved inside a single batch folder. A batch-level
    metadata.json (containing per-image entries) is written at the end of the run.
    Additionally, each generated image is logged as a row in the global CSV:
    outputs/logs/generations.csv
    """
    if style:
        full_prompt = f"{prompt}, {style}"
    else:
        full_prompt = prompt

    # Prepare generator for reproducibility
    device_name = "cpu"
    try:
        device_name = pipe.device.type if hasattr(pipe, "device") else str(pipe.device)
    except Exception:
        device_name = "cpu"

    generator = None
    if seed is not None and int(seed) != 0:
        try:
            generator = torch.Generator(device=device_name).manual_seed(int(seed))
        except Exception:
            try:
                generator = torch.Generator().manual_seed(int(seed))
            except Exception:
                generator = None

    # Prepare output folder for the whole run
    if out_dir is None:
        out_dir = _make_run_folder()
    else:
        os.makedirs(out_dir, exist_ok=True)

    results: List[Dict] = []
    start_all = time.perf_counter()

    # run-level metadata container
    run_meta = {
        "run_id": str(uuid.uuid4()),
        "model_id": getattr(pipe, "model_id", "unknown"),
        "prompt": prompt,
        "full_prompt": full_prompt,
        "negative_prompt": negative_prompt,
        "style": style,
        "guidance_scale": guidance_scale,
        "num_inference_steps": num_inference_steps,
        "seed": seed,
        "created_at": datetime.utcnow().isoformat() + "Z",
        "device": device_name,
        "images": []
    }

    for i in range(n_images):
        # notify start of image
        if progress_callback:
            progress_callback(i / n_images, f"Starting image {i+1}/{n_images}...")

        start = time.perf_counter()

        # per-step callback
        def step_callback(step, timestep, latents):
            if progress_callback:
                # fraction across all images: i + step/num_inference_steps
                fraction = (i + (step / max(1, num_inference_steps))) / max(1, n_images)
                progress_callback(fraction, f"Image {i+1}/{n_images}, step {step+1}/{num_inference_steps}")

        # Pipeline call with callback support (diffusers >=0.20)
        try:
            out = pipe(
                full_prompt,
                negative_prompt=negative_prompt,
                guidance_scale=guidance_scale,
                num_inference_steps=num_inference_steps,
                generator=generator,
                callback=step_callback,
                callback_steps=1
            )
        except TypeError:
            # Older diffusers might not support callback arg; fall back to no callback
            out = pipe(
                full_prompt,
                negative_prompt=negative_prompt,
                guidance_scale=guidance_scale,
                num_inference_steps=num_inference_steps,
                generator=generator
            )

        image = out.images[0]

        # watermark if required
        if add_wm:
            try:
                image = add_watermark(image, text=watermark_text)
            except Exception:
                pass

        # create a consistent filename: image_01.png
        idx = i + 1
        filename_base = f"image_{idx:02d}"
        filename, path, meta_path = save_image_with_meta(
            image=image,
            meta={
                "prompt": prompt,
                "full_prompt": full_prompt,
                "negative_prompt": negative_prompt,
                "style": style,
                "guidance_scale": guidance_scale,
                "num_inference_steps": num_inference_steps,
                "seed": seed,
                "model_id": getattr(pipe, "model_id", "unknown"),
                "generated_at": datetime.utcnow().isoformat() + "Z",
                "format": out_format,
                "index": idx
            },
            out_dir=out_dir,
            img_format=out_format,
            filename=filename_base
        )

        elapsed = time.perf_counter() - start
        img_meta = {
            "filename": filename,
            "path": path,
            "meta_path": meta_path,
            "elapsed": elapsed,
        }
        results.append({"filename": filename, "path": path, "meta": img_meta, "elapsed": elapsed})

        # append per-image info to run metadata (for central metadata.json)
        run_meta["images"].append({
            "index": idx,
            "filename": filename,
            "path": path,
            "meta_path": meta_path,
            "elapsed": elapsed
        })

        # add a row to the global CSV log
        try:
            row = {
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "run_id": run_meta["run_id"],
                "index": idx,
                "filename": filename,
                "path": path,
                "prompt": prompt,
                "negative_prompt": negative_prompt or "",
                "style": style or "",
                "model_id": run_meta.get("model_id", ""),
                "seed": seed if seed is not None else "",
                "steps": num_inference_steps,
                "guidance": guidance_scale,
                "elapsed": round(elapsed, 3)
            }
            _append_to_global_csv(row)
        except Exception:
            pass

        # notify image saved
        if progress_callback:
            fraction = (i + 1) / max(1, n_images)
            avg = sum(r["elapsed"] for r in results) / len(results)
            remaining = int(avg * (n_images - (i + 1)))
            progress_callback(min(max(fraction, 0.0), 1.0), f"Saved image {i+1}/{n_images}. Est. remaining {remaining}s")

    # write central run metadata to out_dir/metadata.json
    try:
        batch_meta_path = os.path.join(out_dir, "metadata.json")
        with open(batch_meta_path, "w", encoding="utf-8") as f:
            json.dump(run_meta, f, ensure_ascii=False, indent=2)
    except Exception:
        pass

    total_elapsed = time.perf_counter() - start_all
    run_meta["total_elapsed"] = total_elapsed

    return results


if __name__ == "__main__":
    # quick smoke test
    try:
        print("Loading pipeline (smoke test)...")
        p = load_pipeline()
        print("Pipeline loaded:", getattr(p, "model_id", "unknown"))
        print("Generating a small sample (this will use your GPU/CPU)...")
        res = generate_images(p, "a cat playing guitar, ultra-detailed", n_images=1, num_inference_steps=20)
        print("Result saved:", res[0]["path"])
    except Exception as e:
        print("Smoke test failed:", e)
