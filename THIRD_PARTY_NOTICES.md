# Third-party notices

HybridMind ports or reimplements mechanisms from the projects below. Each
ported module names its upstream repository, pinned commit, file paths and
licence in its header and lists behavioural deviations in a `DEVIATIONS`
constant. The mapping from upstream mechanism to HybridMind code, with
verification and measured results, is in `docs/REPRODUCTION_MAP.md`.

## Code ported under the MIT licence

| Project | Pinned commit | Copyright | Used in |
|---|---|---|---|
| Entity–Memory Graph (github.com/Sun668/em_graph_memory) | `f020e855be06ac9f33ec888945ff6b305d81cb07` | Copyright (c) 2026 Sun668 | `engine/entity_graph.py`, `engine/entity_extraction.py`, `engine/dense_channel.py`, `engine/fusion.py` (EMG linear fusion) |
| HippoRAG 2 (github.com/OSU-NLP-Group/HippoRAG) | paper code `191f281122e2437743daf1dd2dae23059a14c088`; HEAD `1438aba3fc44ff10573e5a5e1e7cc3c7f9794aff` | Copyright (c) 2025 OSU Natural Language Processing | `engine/entity_graph.py` (personalized-PageRank seeding and walk), `engine/embedding.py` (`nv_embed` query format) |
| LongMemEval (github.com/xiaowu0162/LongMemEval) | `9e0b455f4ef0e2ab8f2e582289761153549043fc` | Copyright (c) 2024 Di Wu | `scripts/reproduce_longmemeval_retrieval.py` (retrieval metrics, flat BM25), imported by `benchmarks/conversational_metrics.py` |
| opsem (github.com/Chrislysen/opsem) | `68186a45882dd85ea66fcc70ee38ba28f6de9a90` | Copyright (c) 2026 Christian Lysenstøen | `engine/fusion.py` (`zscore_fuse`) |

The MIT licence text, which applies to each project above with its own
copyright line:

```
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## Reimplemented from a specification (no code copied)

| Project | Licence | What | Where |
|---|---|---|---|
| Qdrant (github.com/qdrant/qdrant @ `6ab21cac18ebb6f4ae29102c7f8f5cc11affd5de`) | Apache-2.0, Copyright 2026 Qdrant Solutions GmbH | Distribution-based score fusion formula | `engine/fusion.py` (`dbsf_fuse`) |
| LoCoMo evaluation (github.com/snap-research/locomo @ `3eb6f2c585f5e1699204e3c3bdf7adc5c28cb376`) | CC-BY-NC-4.0 | Evidence-recall metric definition only; the non-commercial code is not copied | `benchmarks/conversational_metrics.py` |
| Emergence "Simple" session grouping (blog description; repository has no licence) | none | Session scored by turn-rank DCG | `engine/evidence.py` (`rank_sessions(method="dcg")`) |
| LazyMem neighbour expansion (paper; repository has no licence) | none | ±w same-session expansion | `engine/evidence.py` (`window_units`) |

## Data

- NLTK English stopword list (198 words, NLTK 3.9.4 `nltk_data` corpus),
  vendored in `engine/entity_graph.py` so the EMG tokenizer needs no runtime
  download. NLTK is Apache-2.0 (NLTK Project).
- LoCoMo (CC-BY-NC-4.0) and LongMemEval (MIT) datasets are read from local
  copies for evaluation only and are not redistributed.

Dependencies installed from PyPI (bm25s MIT, rank-bm25 Apache-2.0, faiss MIT,
networkx BSD, scipy BSD, NumPy BSD) keep their own licences and are not
vendored.
