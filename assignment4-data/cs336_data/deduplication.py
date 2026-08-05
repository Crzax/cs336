import hashlib
import os
import random
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

import mmh3


def _line_hash(line: str) -> bytes:
    return hashlib.sha256(line.encode("utf-8")).digest()


def exact_line_deduplication(
    input_files: list[os.PathLike], output_directory: os.PathLike
) -> None:
    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)

    counts: Counter[bytes] = Counter()
    for path in input_files:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                counts[_line_hash(line.rstrip("\n"))] += 1

    for path in input_files:
        path = Path(path)
        out_path = output_directory / path.name
        with open(path, "r", encoding="utf-8") as fin, open(
            out_path, "w", encoding="utf-8"
        ) as fout:
            for line in fin:
                stripped = line.rstrip("\n")
                if counts[_line_hash(stripped)] == 1:
                    fout.write(stripped + "\n")


# ---------------------------------------------------------------------------
# MinHash + LSH fuzzy document deduplication.
#
# Pipeline for each document:
#   1. Normalize the text (lowercase, strip punctuation, normalize whitespace,
#      NFD-normalize, remove accents) to improve recall of near-duplicates.
#   2. Split into word n-grams.
#   3. Compute a MinHash signature of length `num_hashes` by hashing every
#      n-gram with `num_hashes` independent hash functions and taking the
#      per-hash minimum.
#   4. LSH: divide the signature into `num_bands` bands; documents that collide
#      on any band are candidate near-duplicates.
#   5. Verify candidates by computing the *true* n-gram Jaccard similarity on
#      the normalized text; edges above `jaccard_threshold` are duplicates.
#   6. Cluster duplicates with union-find and keep a single random document
#      from each cluster.
# ---------------------------------------------------------------------------


def _normalize(text: str) -> str:
    """Normalize a document to improve near-duplicate recall.

    Applies NFD unicode normalization, removes combining accents, lowercases,
    strips punctuation, and collapses whitespace.
    """
    # NFD unicode normalization.
    text = unicodedata.normalize("NFD", text)
    # Remove combining marks (accents).
    text = "".join(char for char in text if not unicodedata.combining(char))
    # Lowercase.
    text = text.lower()
    # Replace any non-alphanumeric character with a space.
    text = re.sub(r"[^\w\s]", " ", text)
    # Collapse and trim whitespace.
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _word_ngrams(text: str, n: int) -> list[tuple[str, ...]]:
    """Split normalized text into word n-grams."""
    words = text.split()
    if len(words) < n:
        return [tuple(words)]
    return [tuple(words[i : i + n]) for i in range(len(words) - n + 1)]


def _minhash_signature(shingles: list[tuple[str, ...]], num_hashes: int) -> list[int]:
    """Compute a MinHash signature of length `num_hashes` for a set of shingles."""
    signature = [2**32 - 1] * num_hashes
    for shingle in shingles:
        shingle_bytes = " ".join(shingle).encode("utf-8")
        for i in range(num_hashes):
            # Using a distinct seed per index yields independent hash functions.
            value = mmh3.hash(shingle_bytes, seed=i, signed=False)
            if value < signature[i]:
                signature[i] = value
    return signature


def _lsh_bands(
    signature: list[int], num_bands: int
) -> list[tuple[int, int]]:
    """Partition a MinHash signature into bands and hash each band.

    Returns a list of (band_index, band_hash) pairs, one per band.
    """
    rows_per_band = len(signature) // num_bands
    bands: list[tuple[int, int]] = []
    for b in range(num_bands):
        band_values = signature[b * rows_per_band : (b + 1) * rows_per_band]
        band_bytes = repr(band_values).encode("utf-8")
        bands.append((b, mmh3.hash(band_bytes, signed=False)))
    return bands


def _jaccard(a: set, b: set) -> float:
    """Jaccard similarity between two sets."""
    if not a and not b:
        return 1.0
    union = len(a | b)
    if union == 0:
        return 0.0
    return len(a & b) / union


class _UnionFind:
    def __init__(self, n: int):
        self.parent = list(range(n))

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, x: int, y: int) -> None:
        rx, ry = self.find(x), self.find(y)
        if rx != ry:
            self.parent[ry] = rx


def minhash_deduplication(
    input_files: list[os.PathLike],
    output_directory: os.PathLike,
    num_hashes: int,
    num_bands: int,
    ngrams: int,
    jaccard_threshold: float,
    seed: int | None = None,
) -> None:
    """Perform fuzzy document deduplication using MinHash + LSH.

    Args:
        input_files: Paths to input files, one document per file.
        output_directory: Directory where deduplicated documents are written,
            keeping the original file names.
        num_hashes: Number of hash functions for the MinHash signature. Must be
            evenly divisible by `num_bands`.
        num_bands: Number of LSH bands to use.
        ngrams: N-gram length (in words) used for MinHash signatures and
            Jaccard similarity.
        jaccard_threshold: Documents with n-gram Jaccard similarity above this
            threshold are considered duplicates.
        seed: Optional random seed controlling which document of a duplicate
            cluster is retained.
    """
    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)

    paths = [Path(path) for path in input_files]
    num_docs = len(paths)

    # 1. Read and normalize each document.
    normalized: list[str] = []
    for path in paths:
        with open(path, "r", encoding="utf-8") as f:
            normalized.append(_normalize(f.read()))

    # 2. Build n-gram sets (for LSH MinHash and for true Jaccard verification).
    shingle_sets: list[set[tuple[str, ...]]] = []
    signatures: list[list[int]] = []
    for i in range(num_docs):
        shingles = _word_ngrams(normalized[i], ngrams)
        shingle_sets.append(set(shingles))
        signatures.append(_minhash_signature(shingles, num_hashes))

    # 3. LSH: group documents by (band_index, band_hash) to find candidates.
    buckets: dict[tuple[int, int], list[int]] = defaultdict(list)
    for doc_idx in range(num_docs):
        for band_index, band_hash in _lsh_bands(signatures[doc_idx], num_bands):
            buckets[(band_index, band_hash)].append(doc_idx)

    candidate_pairs: set[tuple[int, int]] = set()
    for group in buckets.values():
        for i in range(len(group)):
            for j in range(i + 1, len(group)):
                a, b = group[i], group[j]
                candidate_pairs.add((min(a, b), max(a, b)))

    # 4. Verify candidates using the true n-gram Jaccard similarity.
    uf = _UnionFind(num_docs)
    for a, b in candidate_pairs:
        if _jaccard(shingle_sets[a], shingle_sets[b]) >= jaccard_threshold:
            uf.union(a, b)

    # 5. Group documents into duplicate clusters and keep one random doc each.
    clusters: dict[int, list[int]] = defaultdict(list)
    for doc_idx in range(num_docs):
        clusters[uf.find(doc_idx)].append(doc_idx)

    keep: set[int] = set()
    rng = random.Random(seed)
    for members in clusters.values():
        if len(members) > 1:
            keep.add(rng.choice(members))
        else:
            keep.add(members[0])

    # 6. Write the retained documents to the output directory.
    for doc_idx in range(num_docs):
        if doc_idx not in keep:
            continue
        out_path = output_directory / paths[doc_idx].name
        with open(paths[doc_idx], "r", encoding="utf-8") as fin, open(
            out_path, "w", encoding="utf-8"
        ) as fout:
            fout.write(fin.read())
