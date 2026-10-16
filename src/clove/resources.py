"""clove.resources — build/load the frozen resources (namer, KB, manifest, plan)."""
from __future__ import annotations

import json
from pathlib import Path

from .config import Config
from .data.manifest import (CanonicalManifest, build_manifest, candidates_from_prepared,
                            candidates_from_wacv)
from .data.splits import StreamPlan, eligibility, plan_stream
from .kb.reader import DEFAULT_MENTION_EXEMPT, ClassNamer, KBRegistry, build_registry
from .util import atomic_write_json, read_json


def base_classes(cfg: Config) -> list[str]:
    m = cfg.get("paths.wacv_manifest")
    classes: list[str] = []
    if m and Path(m).exists():
        classes = list(json.loads(Path(m).read_text())["classes"])
    return sorted(set(classes) | set(cfg.get("classes.extra_canonical", []) or []))


def load_namer(cfg: Config) -> ClassNamer:
    return ClassNamer(base_classes(cfg), cfg.get("classes.aliases", {}))


def load_registry(cfg: Config, namer: ClassNamer | None = None) -> tuple[ClassNamer, KBRegistry]:
    namer = namer or load_namer(cfg)
    reg = build_registry(
        namer,
        registry_json=cfg.get("paths.registry_json"),
        provenance_json=cfg.get("paths.provenance_json"),
        kb_xlsx=cfg.get("paths.kb_xlsx"),
        sources_enabled=cfg.get("kb.sources", ["registry_json", "provenance_json", "kb_xlsx"]),
        mention_exempt=tuple(cfg.get("classes.mention_exempt", []) or DEFAULT_MENTION_EXEMPT),
        add_unknown_classes=True,
    )
    return namer, reg


def manifest_path(cfg: Config) -> Path:
    return cfg.artifacts / "manifests" / "manifest.json"


def plan_path(cfg: Config) -> Path:
    return cfg.artifacts / "orders" / "orders.json"


def build_canonical_manifest(cfg: Config, namer: ClassNamer) -> CanonicalManifest:
    cands = []
    if cfg.get("paths.wacv_manifest"):
        cands += candidates_from_wacv(cfg.get("paths.wacv_manifest"), cfg.get("paths.image_dirs", []),
                                      namer, cfg.get("dev.use_manifest_refs", True))
    if cfg.get("paths.prepared_dataset"):
        cands += candidates_from_prepared(cfg.get("paths.prepared_dataset"),
                                          cfg.get("crop").capitalize(), namer)
    if not cands:
        raise FileNotFoundError("no image source configured (paths.wacv_manifest / prepared_dataset)")
    return build_manifest(cands, near_dup_hamming=cfg.get("dedup.near_dup_hamming", 6),
                          image_meta_csv=cfg.get("paths.image_meta_csv"),
                          max_dev_per_class=cfg.get("dev.max_per_class"))


def load_manifest(cfg: Config) -> CanonicalManifest:
    p = manifest_path(cfg)
    if not p.exists():
        raise FileNotFoundError(f"{p} missing — run `clove manifest` first")
    return CanonicalManifest.from_json(read_json(p))


def build_plan(cfg: Config, man: CanonicalManifest, reg: KBRegistry) -> tuple[StreamPlan, dict]:
    elig = eligibility(man.class_counts("test"), reg,
                       min_test_images=cfg.get("classes.min_test_images"),
                       min_claims=cfg.get("classes.min_claims"),
                       hold_out=cfg.get("classes.hold_out", []),
                       exclude=cfg.get("classes.exclude", []))
    plan = plan_stream(elig, reg, cfg["schedule"], cfg["orders"], cfg.get("paths.hard_pairs_csv"))
    return plan, elig.reasons


def load_plan(cfg: Config) -> StreamPlan:
    p = plan_path(cfg)
    if not p.exists():
        raise FileNotFoundError(f"{p} missing — run `clove manifest` first")
    return StreamPlan.from_json(read_json(p))


def freeze(cfg: Config, man: CanonicalManifest, plan: StreamPlan, reasons: dict) -> None:
    mp, pp = manifest_path(cfg), plan_path(cfg)
    if mp.exists() and read_json(mp).get("fingerprint") != man.fingerprint:
        raise RuntimeError(f"{mp} exists with a different fingerprint; refusing to overwrite a "
                           "frozen manifest. Move it aside deliberately to rebuild.")
    if pp.exists() and read_json(pp).get("fingerprint") != plan.fingerprint:
        raise RuntimeError(f"{pp} exists with a different fingerprint; orders are pre-committed.")
    atomic_write_json(mp, man.to_json())
    atomic_write_json(pp, {**plan.to_json(), "eligibility_reasons": reasons})
