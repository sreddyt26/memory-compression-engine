"""
Project.py — AI Digital Memory Compression Engine (PURE PYTHON version)
=======================================================================
Uses ONLY Python's standard library. No numpy, no scikit-learn, no faiss.
Nothing that a Windows "Application Control policy" can block.

Just run:
    python Project.py

Requires nothing extra — only Python itself.
"""

import re
import math
from dataclasses import dataclass, field
from typing import List, Tuple, Dict
from collections import Counter


# ======================================================================
# 1. EMBEDDER  (pure-python TF-IDF -> cosine similarity)
# ======================================================================
class PureEmbedder:
    """Builds a TF-IDF vocabulary over everything it has seen, and encodes
    text into sparse TF-IDF vectors (dicts). Cosine similarity computed by hand."""

    _STOP = {
        "the", "a", "an", "and", "or", "but", "is", "are", "was", "were",
        "to", "of", "in", "on", "for", "we", "i", "it", "that", "this",
        "with", "as", "at", "by", "be", "our", "they", "them", "he", "she",
        "not", "so", "just", "some", "about", "already", "again", "up",
    }

    def __init__(self):
        self.docs_tokens: List[List[str]] = []   # tokenized corpus
        self.df: Counter = Counter()             # document frequency
        self.n_docs = 0

    def _tokenize(self, text: str) -> List[str]:
        toks = re.findall(r"[a-z0-9]+", text.lower())
        return [t for t in toks if t not in self._STOP and len(t) > 1]

    def fit(self, texts: List[str]):
        for t in texts:
            toks = self._tokenize(t)
            self.docs_tokens.append(toks)
            for w in set(toks):
                self.df[w] += 1
            self.n_docs += 1

    def _idf(self, word: str) -> float:
        return math.log((1 + self.n_docs) / (1 + self.df.get(word, 0))) + 1.0

    def vector(self, text: str) -> Dict[str, float]:
        toks = self._tokenize(text)
        if not toks:
            return {}
        tf = Counter(toks)
        total = len(toks)
        vec = {w: (c / total) * self._idf(w) for w, c in tf.items()}
        # L2 normalize
        norm = math.sqrt(sum(v * v for v in vec.values())) or 1.0
        return {w: v / norm for w, v in vec.items()}


def cosine(a: Dict[str, float], b: Dict[str, float]) -> float:
    if not a or not b:
        return 0.0
    # iterate the smaller dict
    if len(a) > len(b):
        a, b = b, a
    return sum(v * b.get(w, 0.0) for w, v in a.items())


# ======================================================================
# 2. COMPRESSOR
# ======================================================================
def split_sentences(text: str) -> List[str]:
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p.strip() for p in parts if len(p.strip()) > 0]


_SALIENT_CUES = {
    "decided", "agreed", "must", "need", "important", "deadline",
    "name", "email", "prefer", "want", "goal", "problem", "because",
    "never", "always", "remember", "note", "key", "budget", "date",
}


def _keyword_score(sentence: str) -> float:
    words = set(re.findall(r"[a-z]+", sentence.lower()))
    hits = len(words & _SALIENT_CUES)
    has_number = 1.0 if re.search(r"\d", sentence) else 0.0
    return hits + 0.5 * has_number


@dataclass
class Memory:
    text: str
    source_text: str
    importance: float
    tier: str = "raw"
    tokens_original: int = 0
    tokens_compressed: int = 0
    meta: dict = field(default_factory=dict)

    @property
    def compression_ratio(self) -> float:
        if self.tokens_original == 0:
            return 1.0
        return self.tokens_compressed / self.tokens_original


def _approx_tokens(text: str) -> int:
    return max(1, int(len(text.split()) / 0.75))


# a shared embedder instance (fit lazily on the sentences we compress)
_EMBEDDER = PureEmbedder()


def extractive_compress(text: str, keep_ratio: float = 0.5) -> str:
    sents = split_sentences(text)
    if len(sents) <= 1:
        return text.strip()

    # local TF-IDF over just these sentences for centrality
    local = PureEmbedder()
    local.fit(sents)
    vecs = [local.vector(s) for s in sents]

    # centroid = average vector
    centroid: Dict[str, float] = {}
    for v in vecs:
        for w, val in v.items():
            centroid[w] = centroid.get(w, 0.0) + val
    ncount = len(vecs) or 1
    centroid = {w: val / ncount for w, val in centroid.items()}
    cnorm = math.sqrt(sum(x * x for x in centroid.values())) or 1.0
    centroid = {w: val / cnorm for w, val in centroid.items()}

    centrality = [cosine(v, centroid) for v in vecs]
    keyword = [_keyword_score(s) for s in sents]
    kmax = max(keyword) or 1.0
    keyword = [k / kmax for k in keyword]

    score = [0.35 * c + 0.65 * k for c, k in zip(centrality, keyword)]
    k = max(1, round(len(sents) * keep_ratio))
    top_idx = sorted(sorted(range(len(sents)), key=lambda i: score[i])[-k:])
    return " ".join(sents[i] for i in top_idx)


def importance_score(text: str) -> float:
    sents = split_sentences(text)
    if not sents:
        return 0.0
    kw = sum(_keyword_score(s) for s in sents) / len(sents)
    length_signal = min(len(sents) / 10.0, 1.0)
    raw = 0.7 * min(kw / 2.0, 1.0) + 0.3 * length_signal
    return round(min(raw, 1.0), 3)


def compress(text: str, mode: str = "extractive", tier: str = "light") -> Memory:
    keep = {"raw": 1.0, "light": 0.6, "heavy": 0.4}.get(tier, 0.6)
    if tier == "raw":
        compressed = text.strip()
    else:
        compressed = extractive_compress(text, keep_ratio=keep)

    return Memory(
        text=compressed,
        source_text=text.strip(),
        importance=importance_score(text),
        tier=tier,
        tokens_original=_approx_tokens(text),
        tokens_compressed=_approx_tokens(compressed),
    )


# ======================================================================
# 3. MEMORY STORE  (hierarchical decay + pure-python retrieval)
# ======================================================================
class MemoryStore:
    def __init__(self, raw_window: int = 3, light_window: int = 6):
        self.memories: List[Memory] = []
        self.raw_window = raw_window
        self.light_window = light_window
        self._embedder = PureEmbedder()
        self._vectors: List[Dict[str, float]] = []

    def add(self, text: str, mode: str = "extractive"):
        mem = compress(text, mode=mode, tier="raw")
        self.memories.append(mem)
        self._apply_decay(mode=mode)
        self._rebuild_index()

    def _apply_decay(self, mode: str):
        n = len(self.memories)
        for i, mem in enumerate(self.memories):
            age = n - 1 - i
            if age < self.raw_window:
                target_tier = "raw"
            elif age < self.raw_window + self.light_window:
                target_tier = "light"
            else:
                target_tier = "heavy"

            if mem.importance >= 0.6 and target_tier == "heavy":
                target_tier = "light"

            if mem.tier != target_tier:
                new_mem = compress(mem.source_text, mode=mode, tier=target_tier)
                new_mem.meta = mem.meta
                self.memories[i] = new_mem

    def _rebuild_index(self):
        # refit embedder over all current memory texts, then vectorize
        self._embedder = PureEmbedder()
        texts = [m.text for m in self.memories]
        self._embedder.fit(texts)
        self._vectors = [self._embedder.vector(t) for t in texts]

    def retrieve(self, query: str, k: int = 3) -> List[Tuple[Memory, float]]:
        if not self.memories:
            return []
        qv = self._embedder.vector(query)
        scored = [(cosine(qv, v), i) for i, v in enumerate(self._vectors)]
        scored.sort(reverse=True)
        k = min(k, len(self.memories))
        return [(self.memories[i], float(s)) for s, i in scored[:k]]

    def stats(self) -> dict:
        if not self.memories:
            return {"count": 0}
        orig = sum(m.tokens_original for m in self.memories)
        comp = sum(m.tokens_compressed for m in self.memories)
        tiers: Dict[str, int] = {}
        for m in self.memories:
            tiers[m.tier] = tiers.get(m.tier, 0) + 1
        return {
            "count": len(self.memories),
            "tokens_original": orig,
            "tokens_compressed": comp,
            "compression_ratio": round(comp / orig, 3) if orig else 1.0,
            "space_saved_pct": round(100 * (1 - comp / orig), 1) if orig else 0.0,
            "tiers": tiers,
        }


# ======================================================================
# 4. EVALUATION
# ======================================================================
def keyword_recall(retrieved_text: str, answer_keywords: List[str]) -> float:
    text = retrieved_text.lower()
    hits = sum(1 for kw in answer_keywords if kw.lower() in text)
    return hits / len(answer_keywords) if answer_keywords else 0.0


def evaluate(store: MemoryStore, probes: List[Dict], k: int = 3,
             threshold: float = 0.5) -> Dict:
    all_memory_text = " ".join(m.text for m in store.memories)
    results, retained, retrieved = [], 0, 0
    for p in probes:
        kept = keyword_recall(all_memory_text, p["answer_keywords"]) >= threshold
        retained += int(kept)

        hits = store.retrieve(p["question"], k=k)
        combined = " ".join(m.text for m, _ in hits)
        got = keyword_recall(combined, p["answer_keywords"]) >= threshold
        retrieved += int(got)

        results.append({"question": p["question"], "retained": kept,
                        "retrieved": got,
                        "top_score": round(hits[0][1], 3) if hits else 0.0})

    stats = store.stats()
    n = len(probes) or 1
    return {
        "fact_retention": round(retained / n, 3),
        "retrieval_accuracy": round(retrieved / n, 3),
        "compression_ratio": stats.get("compression_ratio", 1.0),
        "space_saved_pct": stats.get("space_saved_pct", 0.0),
        "probes_retained": retained,
        "probes_retrieved": retrieved,
        "probes_total": len(probes),
        "detail": results,
    }


# ======================================================================
# 5. SAMPLE DATA
# ======================================================================
CONVERSATION = [
    "Hi, my name is Rishi and I'm planning a product launch. My email is "
    "rishi@example.com and I prefer to be contacted by email, not phone. "
    "This is just some background chit chat about the weather being nice today.",

    "We decided the launch date must be September 15th. The budget is $12,000 "
    "and cannot go higher because finance already approved that exact number. "
    "The main goal is to get 500 signups in the first week.",

    "I talked to the design team. They want to use a blue and white color scheme. "
    "Someone mentioned they had lunch at a nice cafe. The logo needs to be "
    "finalized by August 20th, that is an important deadline.",

    "The marketing channels we agreed on are email newsletters and LinkedIn ads. "
    "We are NOT using paid Google ads because they are too expensive for our budget. "
    "Also the intern said the coffee machine is broken again.",

    "Quick update: the venue for the launch event is the Downtown Convention Center. "
    "It holds 300 people. We need to book catering, and the preferred vendor is "
    "Green Leaf Catering because they handle vegetarian options well.",

    "Final note for now: the CEO wants a demo video ready before launch. "
    "The video must be under 2 minutes. Random aside, traffic was terrible this morning.",
]

PROBES = [
    {"question": "When is the launch date?", "answer_keywords": ["September 15"]},
    {"question": "What is the budget?", "answer_keywords": ["12,000"]},
    {"question": "What is the signup goal?", "answer_keywords": ["500"]},
    {"question": "When must the logo be finalized?", "answer_keywords": ["August 20"]},
    {"question": "Which marketing channels were chosen?",
     "answer_keywords": ["email", "LinkedIn"]},
    {"question": "Where is the launch venue?", "answer_keywords": ["Convention Center"]},
    {"question": "Who is the catering vendor?", "answer_keywords": ["Green Leaf"]},
    {"question": "What is the demo video length limit?", "answer_keywords": ["2 minutes"]},
]


# ======================================================================
# 6. MAIN
# ======================================================================
def main():
    print("=== AI Digital Memory Compression Engine (pure-python) ===\n")

    store = MemoryStore(raw_window=2, light_window=2)

    print("Ingesting conversation turns...")
    for i, turn in enumerate(CONVERSATION):
        store.add(turn)
        print(f"  turn {i+1} added ({len(turn.split())} words)")

    print("\n--- Memory state after ingestion ---")
    for m in store.memories:
        print(f"[{m.tier:5}] imp={m.importance:.2f} "
              f"ratio={m.compression_ratio:.2f} :: {m.text[:90]}...")

    print("\n--- Store stats ---")
    for kk, vv in store.stats().items():
        print(f"  {kk}: {vv}")

    print("\n--- Retrieval demo ---")
    q = "What is the budget and launch date?"
    print(f"Query: {q}")
    for m, s in store.retrieve(q, k=2):
        print(f"  ({s:.3f}) {m.text[:100]}")

    print("\n--- Evaluation ---")
    report = evaluate(store, PROBES, k=3)
    print(f"  Fact retention (compression) : {report['fact_retention']*100:.1f}% "
          f"({report['probes_retained']}/{report['probes_total']})")
    print(f"  Retrieval accuracy (top-k)   : {report['retrieval_accuracy']*100:.1f}% "
          f"({report['probes_retrieved']}/{report['probes_total']})")
    print(f"  Compression ratio            : {report['compression_ratio']}")
    print(f"  Space saved                  : {report['space_saved_pct']}%")
    print("\n  Per-probe (kept=survived compression, got=retrieved top-k):")
    for r in report["detail"]:
        print(f"    kept={int(r['retained'])} got={int(r['retrieved'])}  {r['question']}")


if __name__ == "__main__":
    main()
