"""clove.data.leakage — exact and perceptual duplicate detection, group assignment."""
from __future__ import annotations

from collections import defaultdict
from typing import Iterable


def dhash(path: str, size: int = 8) -> int | None:
    """64-bit difference hash; None if the image cannot be decoded."""
    try:
        from PIL import Image
        with Image.open(path) as im:
            g = im.convert("L").resize((size + 1, size), Image.Resampling.LANCZOS)
            px = list(g.tobytes())
    except Exception:  # noqa: BLE001
        return None
    bits = 0
    for r in range(size):
        row = px[r * (size + 1):(r + 1) * (size + 1)]
        for c in range(size):
            bits = (bits << 1) | (1 if row[c] > row[c + 1] else 0)
    return bits


def hamming(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


class _UF:
    def __init__(self) -> None:
        self.p: dict[str, str] = {}

    def find(self, x: str) -> str:
        self.p.setdefault(x, x)
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[max(ra, rb)] = min(ra, rb)


def duplicate_groups(items: Iterable[tuple[str, str, int | None]], max_hamming: int = 6,
                     ) -> tuple[dict[str, str], list[tuple[str, str, int]]]:
    """items: (image_key, sha256, dhash). Returns (image_key -> group id, direct
    duplicate pairs (key_a, key_b, distance)). Distance -1 marks an exact sha256 match.

    Exact sha256 matches and dHash within `max_hamming` bits are unioned. Up to 6000
    hashed images the comparison is exhaustive (exact); above that, candidates come
    from 4 x 16-bit bands, which only guarantees recall for distances <= 3 and is
    reported as approximate by the audit.
    """
    items = list(items)
    uf = _UF()
    pairs: list[tuple[str, str, int]] = []
    by_sha: dict[str, list[str]] = defaultdict(list)
    for k, sha, _ in items:
        uf.find(k)
        by_sha[sha].append(k)
    for ks in by_sha.values():
        for k in ks[1:]:
            uf.union(ks[0], k)
            pairs.append((ks[0], k, -1))
    hashed = [(k, h) for k, _, h in items if h is not None]
    if max_hamming >= 0 and hashed:
        if len(hashed) <= 6000:
            for i in range(len(hashed)):
                ki, hi = hashed[i]
                for j in range(i + 1, len(hashed)):
                    d = hamming(hi, hashed[j][1])
                    if d <= max_hamming:
                        uf.union(ki, hashed[j][0])
                        pairs.append((ki, hashed[j][0], d))
        else:  # banded candidate generation
            buckets: dict[tuple[int, int], list[int]] = defaultdict(list)
            for idx, (_, h) in enumerate(hashed):
                for band in range(4):
                    buckets[(band, (h >> (16 * band)) & 0xFFFF)].append(idx)
            seen: set[tuple[int, int]] = set()
            for ids in buckets.values():
                for a in range(len(ids)):
                    for b in range(a + 1, len(ids)):
                        if (ids[a], ids[b]) in seen:
                            continue
                        seen.add((ids[a], ids[b]))
                        d = hamming(hashed[ids[a]][1], hashed[ids[b]][1])
                        if d <= max_hamming:
                            uf.union(hashed[ids[a]][0], hashed[ids[b]][0])
                            pairs.append((hashed[ids[a]][0], hashed[ids[b]][0], d))
    return {k: "g" + stable_id(uf.find(k)) for k, _, _ in items}, pairs


def stable_id(key: str) -> str:
    import hashlib
    return hashlib.sha1(key.encode()).hexdigest()[:12]


def cross_split_leaks(records, split_a: str = "test", split_b: str = "dev") -> list[tuple[str, str]]:
    """Pairs (a, b) of images in different splits that share a group id."""
    by_group: dict[str, dict[str, list[str]]] = defaultdict(lambda: defaultdict(list))
    for r in records:
        by_group[r.group_id][r.split].append(r.image_id)
    out = []
    for g in by_group.values():
        for a in g.get(split_a, []):
            for b in g.get(split_b, []):
                out.append((a, b))
    return out


def label_conflicts(records) -> list[str]:
    """Groups (duplicates) whose members carry different class labels."""
    labels: dict[str, set[str]] = defaultdict(set)
    for r in records:
        labels[r.group_id].add(r.class_id)
    return sorted(g for g, s in labels.items() if len(s) > 1)


def conflicting_keys(pairs: list[tuple[str, str, int]], label_of: dict[str, str]) -> set[str]:
    """Keys that are DIRECT duplicates of an image carrying a different label. Only
    the images in such pairs are ambiguous; transitive chains are not penalised."""
    out: set[str] = set()
    for a, b, _ in pairs:
        if label_of.get(a) != label_of.get(b):
            out |= {a, b}
    return out
