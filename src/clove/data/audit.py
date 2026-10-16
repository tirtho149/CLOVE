"""clove.data.audit — Gate 0: verified inventory of what actually exists on disk."""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from ..config import Config
from ..kb.provenance import provenance_ledger
from ..util import environment_report, sha256_file
from .manifest import missing_by_class

MANUSCRIPT_NOTE = ("The SAGE manuscript reports soybean class counts that differ between "
                   "settings; the counts below come from files, not the manuscript.")


def audit(cfg: Config) -> dict:
    from ..resources import build_canonical_manifest, build_plan, load_registry
    paths = cfg["paths"]
    files = {}
    for k in ("wacv_manifest", "kb_xlsx", "provenance_json", "registry_json", "image_meta_csv",
              "hard_pairs_csv"):
        p = paths.get(k)
        files[k] = {"path": p, "exists": bool(p and Path(p).exists()),
                    "sha256": sha256_file(p) if p and Path(p).is_file() else None}
    files["image_dirs"] = [{"path": d, "exists": Path(d).exists(),
                            "n_files": sum(1 for _ in Path(d).iterdir()) if Path(d).exists() else 0}
                           for d in paths.get("image_dirs", [])]
    files["prepared_dataset"] = {"path": paths.get("prepared_dataset"),
                                 "exists": bool(paths.get("prepared_dataset") and Path(paths["prepared_dataset"]).exists())}

    namer, reg = load_registry(cfg)
    man = build_canonical_manifest(cfg, namer)
    plan, reasons = build_plan(cfg, man, reg)

    wacv = {}
    if files["wacv_manifest"]["exists"]:
        m = json.loads(Path(paths["wacv_manifest"]).read_text())
        wacv = {"classes": len(m["classes"]), "num_test": m.get("num_test"),
                "test_per_class_manifest": dict(Counter(c for _, c in m["test"]))}

    classes = sorted(set(man.class_counts("test")) | set(reg.class_ids))
    per_class = {}
    missing = missing_by_class(man.excluded)
    ledger = provenance_ledger(reg)
    for c in classes:
        rec = reg._get_record(c)
        per_class[c] = {
            "test_images_present": man.class_counts("test").get(c, 0),
            "test_images_listed": wacv.get("test_per_class_manifest", {}).get(c),
            "test_missing_files": missing.get(c, {}).get("test", 0),
            "dev_images": man.class_counts("dev").get(c, 0),
            "kb": ledger.get(c, {"n_claims": 0}),
            "disease_type": rec.disease_type if rec else "",
            "affected_parts": list(rec.affected_parts) if rec else [],
            "look_alikes": list(rec.look_alikes) if rec else [],
            "eligibility": reasons.get(c, "not_considered"),
        }
    return {
        "note": MANUSCRIPT_NOTE,
        "files": files,
        "kb_sources": reg.sources,
        "kb_unmapped_names": reg.unmapped,
        "kb_fingerprint": reg.fingerprint,
        "manifest_stats": man.stats,
        "manifest_fingerprint": man.fingerprint,
        "wacv_manifest_summary": {k: v for k, v in wacv.items() if k != "test_per_class_manifest"},
        "per_class": per_class,
        "stream": {"admissible": plan.admissible, "streamed": plan.streamed,
                   "held_back": plan.held_back, "permanent_unknown": plan.permanent_unknown,
                   "sizes": plan.sizes, "departures": plan.departures,
                   "hard_pair_source": plan.hard_pair_source,
                   "orders": [{"order_id": o.order_id, "kind": o.kind, "schedule": o.schedule,
                               "hard_pairs": o.hard_pairs} for o in plan.orders]},
        "source_ood_available": len(man.stats.get("source_datasets", {})) > 1 and bool(paths.get("image_meta_csv")),
        "environment": environment_report(),
        "_objects": (namer, reg, man, plan, reasons),
    }


def render_audit_md(a: dict) -> str:
    L = ["# CLOVE soybean audit (Gate 0)", "", f"> {a['note']}", "", "## Files", ""]
    for k, v in a["files"].items():
        if isinstance(v, list):
            for d in v:
                L.append(f"- image dir `{d['path']}`: exists={d['exists']} files={d['n_files']}")
        else:
            L.append(f"- **{k}**: `{v.get('path')}` exists={v.get('exists')}"
                     + (f" sha256={v['sha256'][:12]}…" if v.get("sha256") else ""))
    s = a["manifest_stats"]
    L += ["", "## Manifest", "",
          f"- WACV manifest: {a['wacv_manifest_summary']}",
          f"- kept images: {s['kept']} (fingerprint `{a['manifest_fingerprint']}`)",
          f"- excluded: {s['excluded_by_reason']}",
          f"- test specimen groups: {s['test_groups']}; dedup exact: {s['dedup_exact']}",
          f"- KB unmapped names (not in canonical vocabulary): {a['kb_unmapped_names'] or 'none'}",
          "", "## Per-class inventory", "",
          "| class | test present / listed | missing | dev | KB claims | url+quote | type | eligibility |",
          "|:--|--:|--:|--:|--:|--:|:--|:--|"]
    for c, r in a["per_class"].items():
        L.append(f"| {c} | {r['test_images_present']} / {r['test_images_listed'] if r['test_images_listed'] is not None else '–'} "
                 f"| {r['test_missing_files']} | {r['dev_images']} | {r['kb'].get('n_claims', 0)} "
                 f"| {r['kb'].get('n_with_url_and_quote', 0)} | {r['disease_type'] or '–'} | {r['eligibility']} |")
    st = a["stream"]
    L += ["", "## Class stream", "",
          f"- admissible: {len(st['admissible'])}; streamed: {len(st['streamed'])}; "
          f"held back (never admitted): {st['held_back'] or 'none'}",
          f"- permanent unknowns (images, no admissible KB): {st['permanent_unknown'] or 'none'}",
          f"- stage sizes: {st['sizes']}; departures from 10→15→20→25: {st['departures'] or 'none'}",
          f"- hard-pair source: `{st['hard_pair_source']}`"]
    for o in st["orders"]:
        L.append(f"- **{o['order_id']}** ({o['kind']}): " + " | ".join(", ".join(b) for b in o["schedule"])
                 + (f"  — pairs {o['hard_pairs']}" if o["hard_pairs"] else ""))
    L += ["", "## Availability flags", "",
          f"- source-OOD: {'AVAILABLE' if a['source_ood_available'] else 'NOT AVAILABLE (no verified independent source metadata)'}",
          "- natural field OOD: NOT AVAILABLE (no site/year/camera metadata in SAGE files)",
          f"- final-stage soybean unknowns: {'available' if (st['held_back'] or st['permanent_unknown']) else 'none (no_eligible_unknowns)'}",
          "", "## Environment", "", "```", json.dumps(a["environment"], indent=2), "```", ""]
    return "\n".join(L)
