"""clove.kb.reader — read the EXISTING SAGE knowledge sources into immutable claims.

Three SAGE formats are supported and merged by canonical class id:

* ``registry_json`` — GitHub SAGE ``disease_registry/outputs/<Crop>/final_registry.json``:
  every field has ``value``, ``url`` and a verbatim ``quote``; includes ``look_alikes``.
* ``provenance_json`` — WACV ``part_index/<crop>_sources.json``: ``{url, quote}`` per class.
* ``kb_xlsx`` — WACV ``wacv2027_kb/<crop>/internet.xlsx`` (the KB SAGE k=0 used): a
  free-text Visual Description per class with no per-sentence URL.

Nothing here crawls, generates or edits KB text. The xlsx is parsed with the
standard library (zip + XML) so no spreadsheet dependency is needed.
"""
from __future__ import annotations

import json
import re
import zipfile
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

from ..contracts import Claim, ClaimKind, ClassRecord
from ..util import sha256_file, stable_hash

_NS = {"m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
_SENT = re.compile(r"(?<=[.!?])\s+(?=[A-Z(\"'])")

# Extra literal phrases (regexes) that are never treated as class mentions. Pathogen
# binomials from the registry are added automatically: "Cercospora sojina" names the
# frogeye pathogen, not the Cercospora-blight class.
DEFAULT_MENTION_EXEMPT: tuple[str, ...] = ()
_BINOMIAL = re.compile(r"\b[A-Z][a-z]+ (?:[a-z]{3,})(?: pv\. [a-z]+)?")


def norm_name(name: str) -> str:
    s = re.sub(r"[\s\-/]+", "_", str(name).strip())
    s = re.sub(r"_+", "_", s).strip("_")
    return s


class ClassNamer:
    """Maps any spelling in any SAGE file to one canonical class id."""

    def __init__(self, canonical: Iterable[str], aliases: dict[str, str] | None = None):
        self.canonical = sorted(set(canonical))
        self._lut: dict[str, str] = {norm_name(c).lower(): c for c in self.canonical}
        for alias, target in (aliases or {}).items():
            tgt = self._lut.get(norm_name(target).lower(), norm_name(target))
            self._lut[norm_name(alias).lower()] = tgt

    def resolve(self, name: str) -> str | None:
        return self._lut.get(norm_name(name).lower())

    def add(self, cid: str) -> str:
        cid = norm_name(cid)
        self._lut.setdefault(cid.lower(), cid)
        if cid not in self.canonical:
            self.canonical = sorted(set(self.canonical) | {cid})
        return self._lut[cid.lower()]

    def surface_forms(self, cid: str) -> list[str]:
        forms = {cid.replace("_", " ")}
        forms |= {k.replace("_", " ") for k, v in self._lut.items() if v == cid}
        return sorted({f.lower() for f in forms if len(f) >= 4}, key=len, reverse=True)


# ── xlsx (stdlib) ─────────────────────────────────────────────────────────────

def read_xlsx_rows(path: str | Path) -> list[list[str]]:
    with zipfile.ZipFile(path) as z:
        shared: list[str] = []
        if "xl/sharedStrings.xml" in z.namelist():
            root = ET.fromstring(z.read("xl/sharedStrings.xml"))
            for si in root.findall("m:si", _NS):
                shared.append("".join(t.text or "" for t in si.iter(f"{{{_NS['m']}}}t")))
        sheets = sorted(n for n in z.namelist() if n.startswith("xl/worksheets/sheet"))
        root = ET.fromstring(z.read(sheets[0]))
    rows: list[list[str]] = []
    for row in root.iter(f"{{{_NS['m']}}}row"):
        cells: dict[int, str] = {}
        for c in row.findall("m:c", _NS):
            ref = c.get("r", "")
            col = 0
            for ch in re.match(r"[A-Z]+", ref).group(0):  # type: ignore[union-attr]
                col = col * 26 + (ord(ch) - 64)
            t = c.get("t")
            if t == "inlineStr":
                val = "".join(x.text or "" for x in c.iter(f"{{{_NS['m']}}}t"))
            else:
                v = c.find("m:v", _NS)
                val = "" if v is None or v.text is None else v.text
                if t == "s" and val:
                    val = shared[int(val)]
            cells[col - 1] = val
        if cells:
            rows.append([cells.get(i, "") for i in range(max(cells) + 1)])
    return rows


def split_sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENT.split(text.strip()) if s.strip()]


@dataclass
class RawKB:
    """Everything read from disk, before canonicalisation into the registry."""
    claims: list[dict] = field(default_factory=list)       # partial claim dicts
    records: dict[str, dict] = field(default_factory=dict)  # raw name -> fields
    sources: dict[str, str] = field(default_factory=dict)   # doc label -> sha256


def read_registry_json(path: str | Path, raw: RawKB) -> None:
    data = json.loads(Path(path).read_text())
    raw.sources["registry_json"] = sha256_file(path)
    for e in data.get("diseases", []):
        name = e.get("disease_name")
        if not name:
            continue
        rec = raw.records.setdefault(name, {})
        for fld, label in (("pathogen_scientific_name", "Pathogen"),
                           ("type_of_disease", "Disease type"),
                           ("affected_parts", "Affected parts")):
            f = e.get(fld) or {}
            val = f.get("value")
            if not val:
                continue
            sval = ", ".join(val) if isinstance(val, list) else str(val)
            rec[fld] = sval
            raw.claims.append(dict(name=name, kind=ClaimKind.FIELD, field=fld,
                                   text=f"{label}: {sval}", quote=f.get("quote", ""),
                                   url=f.get("url", ""), doc="registry_json"))
        vs = e.get("visual_symptoms") or {}
        for sub in ("summary", "diagnostic_features"):
            f = vs.get(sub) or {}
            if not f.get("value"):
                continue
            for sent in split_sentences(str(f["value"])):
                raw.claims.append(dict(name=name, kind=ClaimKind.QUOTE,
                                       field=f"visual_symptoms.{sub}", text=sent,
                                       quote=f.get("quote", ""), url=f.get("url", ""),
                                       doc="registry_json"))
        la = vs.get("look_alikes") or {}
        if la.get("value"):
            vals = la["value"] if isinstance(la["value"], list) else [la["value"]]
            rec["look_alikes_raw"] = [str(v) for v in vals]
            raw.claims.append(dict(name=name, kind=ClaimKind.LOOK_ALIKE,
                                   field="visual_symptoms.look_alikes",
                                   text="Documented look-alikes: " + ", ".join(map(str, vals)),
                                   quote=la.get("quote", ""), url=la.get("url", ""),
                                   doc="registry_json"))


def read_provenance_json(path: str | Path, raw: RawKB) -> None:
    data = json.loads(Path(path).read_text())
    raw.sources["provenance_json"] = sha256_file(path)
    for name, items in (data.get("provenance") or {}).items():
        raw.records.setdefault(name, {})
        for it in items or []:
            q = (it or {}).get("quote")
            if q:
                raw.claims.append(dict(name=name, kind=ClaimKind.QUOTE, field="provenance",
                                       text=q.strip(), quote=q.strip(),
                                       url=it.get("url", ""), doc="provenance_json"))


def read_kb_xlsx(path: str | Path, raw: RawKB) -> None:
    rows = read_xlsx_rows(path)
    raw.sources["kb_xlsx"] = sha256_file(path)
    if not rows:
        return
    header = [h.strip().lower() for h in rows[0]]

    def col(row: list[str], name: str) -> str:
        return row[header.index(name)].strip() if name in header and header.index(name) < len(row) else ""

    for row in rows[1:]:
        name = col(row, "disease")
        if not name:
            continue
        rec = raw.records.setdefault(name, {})
        for key, fld in (("pathogen", "pathogen_scientific_name"),
                         ("type of disease", "type_of_disease"),
                         ("affected parts", "affected_parts")):
            v = col(row, key)
            if v and v not in ("—", "-"):
                rec.setdefault(fld, v)
        rec["xlsx_confidence"] = col(row, "confidence")
        desc = col(row, "visual description")
        if desc and desc not in ("—", "-"):
            for sent in split_sentences(desc):
                raw.claims.append(dict(name=name, kind=ClaimKind.KB_DESCRIPTION,
                                       field="visual_description", text=sent, quote="",
                                       url="", doc="kb_xlsx"))


class KBRegistry:
    """Immutable, complete registry. Only `kb.inventory.AuthorizedKB` may read it."""

    def __init__(self, claims: dict[str, tuple[Claim, ...]], records: dict[str, ClassRecord],
                 sources: dict[str, str], unmapped: dict[str, int]):
        self._claims = claims
        self._records = records
        self.sources = dict(sources)
        self.unmapped = dict(unmapped)
        self.fingerprint = stable_hash({"sources": sources,
                                        "claims": {k: [c.claim_id for c in v] for k, v in claims.items()}})

    @property
    def class_ids(self) -> list[str]:
        return sorted(set(self._claims) | set(self._records))

    def n_claims(self, cid: str) -> int:
        return len(self._claims.get(cid, ()))

    def kind_counts(self, cid: str) -> dict[str, int]:
        out: dict[str, int] = {}
        for c in self._claims.get(cid, ()):
            out[c.kind.value] = out.get(c.kind.value, 0) + 1
        return out

    # private accessors — used by AuthorizedKB and by evaluator-only audit code
    def _get_claims(self, cid: str) -> tuple[Claim, ...]:
        return self._claims.get(cid, ())

    def _get_record(self, cid: str) -> ClassRecord | None:
        return self._records.get(cid)


def _find_mentions(text: str, self_id: str, namer: ClassNamer,
                   exempt: list[re.Pattern]) -> tuple[str, ...]:
    masked = text
    for pat in exempt:
        masked = pat.sub(" ", masked)
    low = f" {masked.lower()} "
    hits = []
    for cid in namer.canonical:
        if cid == self_id:
            continue
        for form in namer.surface_forms(cid):
            if re.search(rf"(?<![a-z]){re.escape(form)}(?![a-z])", low):
                hits.append(cid)
                break
    return tuple(sorted(hits))


def build_registry(namer: ClassNamer, *, registry_json: str | None = None,
                   provenance_json: str | None = None, kb_xlsx: str | None = None,
                   sources_enabled: Iterable[str] = ("registry_json", "provenance_json", "kb_xlsx"),
                   mention_exempt: Iterable[str] = DEFAULT_MENTION_EXEMPT,
                   add_unknown_classes: bool = False) -> KBRegistry:
    raw = RawKB()
    enabled = set(sources_enabled)
    if registry_json and "registry_json" in enabled and Path(registry_json).exists():
        read_registry_json(registry_json, raw)
    if provenance_json and "provenance_json" in enabled and Path(provenance_json).exists():
        read_provenance_json(provenance_json, raw)
    if kb_xlsx and "kb_xlsx" in enabled and Path(kb_xlsx).exists():
        read_kb_xlsx(kb_xlsx, raw)

    unmapped: dict[str, int] = {}

    def canon(name: str) -> str | None:
        cid = namer.resolve(name)
        if cid is None and add_unknown_classes:
            cid = namer.add(name)
        if cid is None:
            unmapped[name] = unmapped.get(name, 0) + 1
        return cid

    pathogen_phrases = set()
    for rec in raw.records.values():
        pathogen_phrases.update(_BINOMIAL.findall(str(rec.get("pathogen_scientific_name", ""))))
    exempt = [re.compile(p) for p in mention_exempt]
    exempt += [re.compile(re.escape(b)) for b in sorted(pathogen_phrases, key=len, reverse=True)]
    by_class: dict[str, list[Claim]] = {}
    seen: set[tuple[str, str]] = set()
    for c in raw.claims:
        cid = canon(c["name"])
        if cid is None:
            continue
        text = re.sub(r"\s+", " ", c["text"]).strip()
        if (cid, text.lower()) in seen:       # identical text from two files: keep first
            continue
        seen.add((cid, text.lower()))
        claim_id = f"{cid[:12]}-{stable_hash([cid, c['doc'], c['field'], text], 8)}"
        by_class.setdefault(cid, []).append(Claim(
            claim_id=claim_id, class_id=cid, kind=c["kind"], text=text,
            verbatim_quote=c["quote"] or "", source_url=c["url"] or "", source_doc=c["doc"],
            field=c["field"], mentions=_find_mentions(text, cid, namer, exempt)))

    records: dict[str, ClassRecord] = {}
    merged: dict[str, dict] = {}
    for name, rec in raw.records.items():
        cid = canon(name)
        if cid is None:
            continue
        m = merged.setdefault(cid, {"aliases": set()})
        m["aliases"].add(name)
        for k, v in rec.items():
            m.setdefault(k, v)
    for cid, m in merged.items():
        la = []
        for x in m.get("look_alikes_raw", []):
            for part in re.split(r",|;| and ", x):
                r = namer.resolve(part)
                if r and r != cid:
                    la.append(r)
        parts = tuple(p.strip().lower() for p in re.split(r"[,;/]", m.get("affected_parts", "")) if p.strip())
        records[cid] = ClassRecord(
            class_id=cid, display_name=cid.replace("_", " "),
            aliases=tuple(sorted(m["aliases"])), disease_type=m.get("type_of_disease", ""),
            affected_parts=parts, pathogen=m.get("pathogen_scientific_name", ""),
            look_alikes=tuple(sorted(set(la))))
    return KBRegistry({k: tuple(v) for k, v in by_class.items()}, records, raw.sources, unmapped)
