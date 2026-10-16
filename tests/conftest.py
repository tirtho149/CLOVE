"""Synthetic fixtures for CPU-only tests (Gate 2).

Everything here is DUMMY data written to a pytest tmp dir. It never touches the real
SAGE files and can never enter an evaluation: the fixture classes are named
`Dummy_*` and the artifacts live under tmp_path.
"""
from __future__ import annotations

import hashlib
import json
import random
import zipfile
from pathlib import Path

import pytest

CLASSES = ["Dummy_Alpha_Spot", "Dummy_Beta_Blight", "Dummy_Gamma_Rot", "Dummy_Delta_Mosaic",
           "Dummy_Epsilon_Rust"]
NO_KB = "Dummy_Zeta_Injury"          # images, no KB -> permanent unknown

DESCR = {
    "Dummy_Alpha_Spot": "Small circular tan spots with purple margins appear on upper leaves. "
                        "Unlike Dummy Beta Blight, the spots stay round.",
    "Dummy_Beta_Blight": "Angular water-soaked lesions turn brown with yellow halos on young leaves.",
    "Dummy_Gamma_Rot": "Brown discoloration of the stem pith near the soil line; plants wilt.",
    "Dummy_Delta_Mosaic": "Mottled light and dark green mosaic on leaves with puckering.",
    "Dummy_Epsilon_Rust": "Tan to reddish-brown pustules on lower leaf surfaces release spores.",
}


def write_xlsx(path: Path, rows: list[list[str]]) -> None:
    def cell(ref, v):
        v = v.replace("&", "&amp;").replace("<", "&lt;")
        return f'<c r="{ref}" t="inlineStr"><is><t>{v}</t></is></c>'
    sheet_rows = []
    for i, row in enumerate(rows, 1):
        cells = "".join(cell(f"{chr(65 + j)}{i}", v) for j, v in enumerate(row))
        sheet_rows.append(f'<row r="{i}">{cells}</row>')
    sheet = ('<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
             f'<sheetData>{"".join(sheet_rows)}</sheetData></worksheet>')
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("xl/worksheets/sheet1.xml", sheet)
        z.writestr("xl/workbook.xml", "<workbook/>")


def make_image(path: Path, seed: int) -> None:
    from PIL import Image
    rng = random.Random(seed)
    im = Image.new("RGB", (32, 32))
    px = im.load()
    for x in range(32):
        for y in range(32):
            px[x, y] = (rng.randrange(256), rng.randrange(256), rng.randrange(256))
    im.save(path, format="JPEG", quality=95)


@pytest.fixture()
def sage_fixture(tmp_path: Path) -> dict:
    img_dir = tmp_path / "pool"
    img_dir.mkdir()
    test, refs, ext = [], {}, {}
    seed = 0
    for c in CLASSES + [NO_KB]:
        refs[c] = []
        for i in range(8):
            seed += 1
            p = img_dir / f"tmp{seed}.jpg"
            make_image(p, seed)
            sha = hashlib.sha1(p.read_bytes()).hexdigest()
            p.rename(img_dir / f"{sha}.jpg")
            ext[sha] = ".jpg"
            if i < 5:
                test.append([sha, c])
            else:
                refs[c].append(sha)
    manifest = {"crop": "soybean", "classes": CLASSES + [NO_KB], "ext": ext, "test": test,
                "ref": {"internet": refs}, "num_test": len(test)}
    mpath = tmp_path / "manifest.json"
    mpath.write_text(json.dumps(manifest))
    rows = [["Disease", "Pathogen", "Type of Disease", "Affected Parts", "Visual Description",
             "Confidence", "Num Sources"]]
    types = ["Fungal", "Bacterial", "Fungal", "Viral", "Fungal"]
    parts = ["Foliar", "Foliar", "Stem, Root", "Foliar", "Foliar"]
    for c, t, pa in zip(CLASSES, types, parts):
        rows.append([c, "Fakeus dummyi", t, pa, DESCR[c], "high", "2"])
    xlsx = tmp_path / "internet.xlsx"
    write_xlsx(xlsx, rows)
    prov = {"crop": "Soybean", "provenance": {c: [{"url": f"https://example.edu/{c.lower()}",
                                                    "quote": DESCR[c].split(". ")[0] + "."}] for c in CLASSES}}
    ppath = tmp_path / "sources.json"
    ppath.write_text(json.dumps(prov))
    reg = {"diseases": [{"disease_name": "Dummy_Alpha_Spot",
                         "affected_parts": {"value": ["Foliar"], "url": "https://example.edu/a", "quote": "leaves"},
                         "visual_symptoms": {"look_alikes": {"value": ["Dummy Beta Blight"],
                                                             "url": "https://example.edu/a", "quote": "look alike"}}}]}
    rpath = tmp_path / "registry.json"
    rpath.write_text(json.dumps(reg))
    cfg = {
        "paths": {"wacv_manifest": str(mpath), "kb_xlsx": str(xlsx), "provenance_json": str(ppath),
                  "registry_json": str(rpath), "image_dirs": [str(img_dir)],
                  "artifacts": str(tmp_path / "artifacts")},
        "classes": {"min_test_images": 3, "min_claims": 1},
        "schedule": {"initial": 2, "batch": 1, "stages": 4, "min_initial": 2, "min_batch": 1},
        "orders": {"random_orders": 2, "seed": 7},
        "retrieval": {"visual": "none"},
        "model": {"backend": "openai_compat", "require_exact_tokenizer": False},
        "gate": {"dev_per_class": 3},
        "stats": {"bootstrap": 200},
        "methods": ["names", "sage0", "append", "fresh", "prism"],
        "smoke": {"classes": 3, "images_per_class": 2, "dev_per_class": 2},
    }
    cpath = tmp_path / "pilot.yaml"
    import yaml
    cpath.write_text(yaml.safe_dump(cfg))
    return {"root": tmp_path, "config": str(cpath), "cfg": cfg, "manifest": manifest}
