"""clove.data.transforms — pre-registered, deterministic covariate shifts.

Variants are derived from the SAME original test image (paired design) and inherit
its group id, so a variant can never act as an independent specimen. Severities are
fixed in config before any results are inspected.
"""
from __future__ import annotations

import io
from dataclasses import replace
from pathlib import Path

from ..contracts import ImageRecord
from ..util import sha256_file

# shift_type -> (kind, description). 'label_preserving' vs 'evidence_loss' is declared
# up front, never decided after looking at accuracy.
SHIFTS = {
    "brightness": "label_preserving",
    "contrast": "label_preserving",
    "blur": "label_preserving",          # mild sigma only; severe blur is evidence_loss
    "jpeg": "label_preserving",
    "downscale": "label_preserving",
    "white_balance": "label_preserving",
    "severe_blur": "evidence_loss",
    "occlusion": "evidence_loss",
    "center_crop": "evidence_loss",
}


def apply_shift(src: str, shift: str, severity: float, dst: str) -> None:
    from PIL import Image, ImageDraw, ImageEnhance, ImageFilter
    with Image.open(src) as im0:
        im = im0.convert("RGB")
    if shift == "brightness":
        im = ImageEnhance.Brightness(im).enhance(severity)
    elif shift == "contrast":
        im = ImageEnhance.Contrast(im).enhance(severity)
    elif shift in ("blur", "severe_blur"):
        im = im.filter(ImageFilter.GaussianBlur(radius=severity))
    elif shift == "jpeg":
        buf = io.BytesIO()
        im.save(buf, format="JPEG", quality=int(severity))
        buf.seek(0)
        im = Image.open(buf).convert("RGB")
    elif shift == "downscale":
        w, h = im.size
        small = im.resize((max(8, int(w * severity)), max(8, int(h * severity))), Image.Resampling.BILINEAR)
        im = small.resize((w, h), Image.Resampling.BILINEAR)
    elif shift == "white_balance":
        r, g, b = im.split()
        r = r.point(lambda v: min(255, int(v * severity)))
        b = b.point(lambda v: min(255, int(v / severity)))
        im = Image.merge("RGB", (r, g, b))
    elif shift == "occlusion":
        w, h = im.size
        side = int(min(w, h) * severity)
        d = ImageDraw.Draw(im)
        d.rectangle([(w - side) // 2, (h - side) // 2, (w + side) // 2, (h + side) // 2], fill=(128, 128, 128))
    elif shift == "center_crop":
        w, h = im.size
        cw, ch = int(w * severity), int(h * severity)
        im = im.crop(((w - cw) // 2, (h - ch) // 2, (w + cw) // 2, (h + ch) // 2))
    else:
        raise ValueError(f"unknown shift {shift}")
    Path(dst).parent.mkdir(parents=True, exist_ok=True)
    im.save(dst, format="JPEG", quality=95)


def make_variants(test: list[ImageRecord], spec: dict[str, list[float]], out_dir: str
                  ) -> list[ImageRecord]:
    """spec: {shift: [severity, ...]} from config.shifts.covariate."""
    out: list[ImageRecord] = []
    for r in test:
        if r.split != "test":
            raise ValueError("variants are only generated from held-out test originals")
        for shift, sevs in spec.items():
            if shift not in SHIFTS:
                raise ValueError(f"shift {shift} is not pre-registered")
            for s in sevs:
                vid = f"{shift}@{s:g}"
                dst = str(Path(out_dir) / r.image_id / f"{vid}.jpg")
                if not Path(dst).exists():
                    apply_shift(r.path, shift, s, dst)
                out.append(replace(r, path=dst, variant_id=vid, shift_type=f"covariate:{shift}",
                                   severity=float(s), sha256=sha256_file(dst)))
    return out
