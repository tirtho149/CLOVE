"""clove.retrieval.embedding — image→class-text scores s_visual(x, c).

This is NOT a VLM call and uses NO reference images: it scores the query photo against
the TEXT of admitted classes (name + affected parts + top card claims) with a frozen
dual encoder (SigLIP). Image embeddings are cached by image sha256; text embeddings by
(class, card hash). `precomputed` loads a JSON {image_id: {class_id: score}} produced
offline with the same rules.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Protocol

from ..contracts import AccessScope


class VisualScorer(Protocol):
    name: str

    def scores(self, image_path: str, image_sha: str, class_texts: dict[str, str],
               scope: AccessScope) -> dict[str, float]: ...


class NoScorer:
    name = "none"

    def scores(self, image_path, image_sha, class_texts, scope):  # noqa: D401
        return {}


class PrecomputedScorer:
    name = "precomputed"

    def __init__(self, path: str):
        self.table: dict[str, dict[str, float]] = json.loads(Path(path).read_text())

    def scores(self, image_path, image_sha, class_texts, scope):
        row = self.table.get(image_sha) or {}
        return {c: float(row[c]) for c in scope.allowed_class_ids if c in row}


class SiglipScorer:
    def __init__(self, model_name: str, device: str = "auto", cache_dir: str | None = None):
        import torch
        from transformers import AutoModel, AutoProcessor
        self.torch = torch
        self.device = ("cuda" if torch.cuda.is_available() else "cpu") if device == "auto" else device
        self.model = AutoModel.from_pretrained(model_name).to(self.device).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.proc = AutoProcessor.from_pretrained(model_name)
        self.name = f"siglip:{model_name}"
        self._img: dict[str, object] = {}
        self._txt: dict[str, object] = {}

    def _image(self, path: str, sha: str):
        if sha not in self._img:
            from PIL import Image
            with Image.open(path) as im:
                inp = self.proc(images=im.convert("RGB"), return_tensors="pt").to(self.device)
            with self.torch.inference_mode():
                e = self.model.get_image_features(**inp)
            self._img[sha] = e / e.norm(dim=-1, keepdim=True)
        return self._img[sha]

    def _text(self, key: str, text: str):
        if key not in self._txt:
            inp = self.proc(text=[text], padding="max_length", truncation=True,
                            return_tensors="pt").to(self.device)
            with self.torch.inference_mode():
                e = self.model.get_text_features(**inp)
            self._txt[key] = e / e.norm(dim=-1, keepdim=True)
        return self._txt[key]

    def scores(self, image_path, image_sha, class_texts, scope):
        ie = self._image(image_path, image_sha)
        out = {}
        for c, t in class_texts.items():
            scope.check(c, "visual scoring of")
            te = self._text(f"{c}::{hash(t)}", t)
            out[c] = float((ie @ te.T).item())
        return out


def make_scorer(cfg_retrieval: dict) -> VisualScorer:
    kind = cfg_retrieval.get("visual", "none")
    if kind == "none":
        return NoScorer()
    if kind == "precomputed":
        return PrecomputedScorer(cfg_retrieval["precomputed_scores"])
    if kind == "siglip":
        return SiglipScorer(cfg_retrieval["siglip_model"], cfg_retrieval.get("device", "auto"))
    raise ValueError(f"unknown visual retriever {kind}")
