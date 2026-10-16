"""clove.data.manifest — the canonical, frozen image manifest.

Image sources (both read-only):
* the WACV frozen manifest (``data/manifest/<crop>.json``): content-addressed test ids
  plus per-class ``ref`` ids. Under CLOVE the ``ref`` images are NEVER shown to the model
  (k_img = 0); they become the *development* split used only for unknown-gate
  calibration and for development-time confusion priors.
* optionally the GitHub SAGE ``Prepared_Dataset/<Crop>{,_test}/<Class>/<part>/*`` tree.

Duplicates are grouped (exact sha256 + dHash). A dev image that shares a group with
any test image is dropped so test images stay untouched; groups with conflicting
labels are excluded from both splits and reported.
"""
from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path

from ..contracts import ImageRecord
from ..kb.reader import ClassNamer
from ..util import sha256_file, stable_hash
from .leakage import conflicting_keys, dhash, duplicate_groups

IMG_EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


@dataclass
class Candidate:
    key: str
    image_id: str
    path: str | None
    raw_label: str
    class_id: str | None
    split: str
    source_dataset: str
    organ: str = ""


@dataclass
class CanonicalManifest:
    images: list[ImageRecord]
    excluded: list[dict] = field(default_factory=list)
    stats: dict = field(default_factory=dict)
    fingerprint: str = ""

    def to_json(self) -> dict:
        return {"fingerprint": self.fingerprint, "stats": self.stats,
                "excluded": self.excluded, "images": [asdict(r) for r in self.images]}

    @classmethod
    def from_json(cls, d: dict) -> "CanonicalManifest":
        return cls([ImageRecord(**r) for r in d["images"]], d.get("excluded", []),
                   d.get("stats", {}), d.get("fingerprint", ""))

    def by_split(self, split: str) -> list[ImageRecord]:
        return [r for r in self.images if r.split == split]

    def class_counts(self, split: str) -> dict[str, int]:
        return dict(Counter(r.class_id for r in self.images if r.split == split))


def _locate(image_id: str, ext: str, image_dirs: list[str]) -> str | None:
    for d in image_dirs:
        p = Path(d) / f"{image_id}{ext}"
        if p.exists():
            return str(p)
    return None


def candidates_from_wacv(manifest_path: str, image_dirs: list[str], namer: ClassNamer,
                         use_refs: bool = True) -> list[Candidate]:
    m = json.loads(Path(manifest_path).read_text())
    ext = m.get("ext", {})
    if isinstance(ext, list):        # some copies store only ids; assume .jpg
        ext = {k: ".jpg" for k in ext}
    out: list[Candidate] = []
    for i, (sha, label) in enumerate(m["test"]):
        out.append(Candidate(f"wacv:test:{i}", sha, _locate(sha, ext.get(sha, ".jpg"), image_dirs),
                             label, namer.resolve(label), "test", "sage_wacv_pool"))
    if use_refs:
        refs = (m.get("ref") or {}).get("internet") or {}
        for label, shas in refs.items():
            for j, sha in enumerate(shas):
                out.append(Candidate(f"wacv:ref:{label}:{j}", sha,
                                     _locate(sha, ext.get(sha, ".jpg"), image_dirs),
                                     label, namer.resolve(label), "dev", "sage_wacv_pool"))
    return out


def candidates_from_prepared(root: str, crop_dir: str, namer: ClassNamer) -> list[Candidate]:
    out: list[Candidate] = []
    for split, sub in (("test", f"{crop_dir}_test"), ("dev", crop_dir)):
        base = Path(root) / sub
        if not base.exists():
            continue
        for p in sorted(base.rglob("*")):
            if p.suffix.lower() not in IMG_EXT or "rejected" in p.parts:
                continue
            rel = p.relative_to(base).parts
            label = rel[0]
            organ = rel[1] if len(rel) > 2 else ""
            out.append(Candidate(f"prep:{split}:{p.relative_to(base)}", "", str(p), label,
                                 namer.resolve(label), split, f"sage_prepared_{split}", organ))
    return out


def _read_meta(path: str | None) -> dict[str, dict]:
    if not path or not Path(path).exists():
        return {}
    with open(path) as f:
        return {row["image_id"]: row for row in csv.DictReader(f)}


def build_manifest(cands: list[Candidate], *, near_dup_hamming: int = 6,
                   image_meta_csv: str | None = None, max_dev_per_class: int | None = None,
                   ) -> CanonicalManifest:
    excluded: list[dict] = []
    present: list[tuple[Candidate, str, int | None]] = []
    for c in cands:
        if c.class_id is None:
            excluded.append({"key": c.key, "reason": "unmapped_label", "label": c.raw_label})
            continue
        if c.path is None:
            excluded.append({"key": c.key, "reason": "file_missing", "class_id": c.class_id,
                             "split": c.split, "image_id": c.image_id})
            continue
        h = dhash(c.path)
        if h is None:
            excluded.append({"key": c.key, "reason": "decode_failed", "class_id": c.class_id})
            continue
        present.append((c, sha256_file(c.path), h))

    groups, pairs = duplicate_groups([(c.key, sha, h) for c, sha, h in present], near_dup_hamming)
    conflicts = conflicting_keys(pairs, {c.key: c.class_id or "" for c, _, _ in present})
    meta = _read_meta(image_meta_csv)

    def rec(c: Candidate, sha: str) -> ImageRecord:
        iid = c.image_id or sha[:40]
        m = meta.get(iid, {})
        return ImageRecord(image_id=iid, path=c.path or "", class_id=c.class_id or "",
                           split=c.split, source_dataset=m.get("source_dataset") or c.source_dataset,
                           # chained components can span labels; the class suffix keeps a
                           # group single-label (cross-label direct pairs are excluded below)
                           group_id=m.get("group_id") or f"{groups[c.key]}:{c.class_id}", sha256=sha)

    recs = [(c, rec(c, sha)) for c, sha, _ in present]
    test_groups = {r.group_id for c, r in recs if r.split == "test" and c.key not in conflicts}

    images: list[ImageRecord] = []
    seen_test_keys: set[tuple[str, str]] = set()
    dev_per_class: Counter = Counter()
    for c, r in recs:
        if c.key in conflicts:
            excluded.append({"key": c.key, "reason": "cross_label_duplicate",
                             "group_id": r.group_id, "class_id": r.class_id, "split": r.split})
            continue
        if r.split == "dev":
            if r.group_id in test_groups:
                excluded.append({"key": c.key, "reason": "dev_duplicates_test",
                                 "group_id": r.group_id, "class_id": r.class_id})
                continue
            if max_dev_per_class and dev_per_class[r.class_id] >= max_dev_per_class:
                continue
            dev_per_class[r.class_id] += 1
        else:
            # the WACV pool stores byte-identical test images once; same id twice in the
            # test list is ONE specimen — keep one evaluation instance
            if (r.image_id, r.class_id) in seen_test_keys:
                excluded.append({"key": c.key, "reason": "repeated_test_id", "image_id": r.image_id})
                continue
            seen_test_keys.add((r.image_id, r.class_id))
        images.append(r)

    images.sort(key=lambda r: (r.split, r.class_id, r.image_id))
    stats = {
        "candidates": len(cands),
        "kept": len(images),
        "excluded_by_reason": dict(Counter(e["reason"] for e in excluded)),
        "test_per_class": dict(sorted(Counter(r.class_id for r in images if r.split == "test").items())),
        "dev_per_class": dict(sorted(Counter(r.class_id for r in images if r.split == "dev").items())),
        "test_groups": len({r.group_id for r in images if r.split == "test"}),
        "near_dup_hamming": near_dup_hamming,
        "duplicate_pairs": len(pairs),
        "dedup_exact": len(present) <= 6000,
        "source_datasets": dict(Counter(r.source_dataset for r in images)),
    }
    fp = stable_hash([(r.image_id, r.sha256, r.class_id, r.split, r.group_id) for r in images])
    return CanonicalManifest(images, excluded, stats, fp)


def missing_by_class(excluded: list[dict]) -> dict[str, dict[str, int]]:
    out: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for e in excluded:
        if e["reason"] == "file_missing":
            out[e["class_id"]][e["split"]] += 1
    return {k: dict(v) for k, v in out.items()}
