# Research brief: long-term conversational memory on LongMemEval and LoCoMo, as of 2026-09-25

**Provenance.** This brief is built from 5 source sweeps and 3 adversarial verification passes. Today I also ran 3 web searches and 8 web fetches, aimed at the novelty question. No local compute was run, no datasets were downloaded, and no LLM or provider calls were made. The fetch tool cached one paper PDF (PACMS, about 710 KB) on its own.

**Verdict codes**
- **V**: checked against the primary paper, repo or page.
- **A**: recounted from released per-question files.
- **C**: the headline number is confirmed, but a detail was corrected (see the note in the row).
- **S**: secondary source, or a rerun by a competitor.
- **U**: cannot be verified.
- **N**: new in this pass. Taken from a summarising fetch of arXiv HTML, not checked against the page image.

**Abbreviations**
- off-J: the official LongMemEval judge (`evaluate_qa.py`): `gpt-4o-2024-08-06`, temperature 0, max_tokens 10, a "yes" substring check, per-type prompts plus an abstention prompt.
- abs: the 30 abstention questions (`_abs`).
- S orig / S cln: the original 2024 release vs the September 2025 cleaned release.
- J: binary LLM-judge accuracy.
- Mem0-gen: the Mem0 judge prompt ("be generous … touches on the same topic").
- FC: full-context baseline.
- rot: category labels follow the Mem0-paper rotation (1=single, 2=temporal, 3=multi, 4=open), which is wrong. The official code mapping is 1=multi-hop (282), 2=temporal (321), 3=open-domain (96), 4=single-hop (841), 5=adversarial (446).

---

## 1. Published claims, grouped by what the metric measures

### 1A. LongMemEval: answer accuracy on the full benchmark (n=500 including 30 abs unless noted)

| System [src] | Data | Value | Denominator | Reader | Judge | Budget | Cost/latency | Code (licence) | Per-question artifacts | Verdict |
|---|---|---|---|---|---|---|---|---|---|---|
| LongMemEval paper, FC and oracle [LME] | S orig; oracle | GPT-4o on S: 60.6 (64.0 with CoN). Oracle: 87.0 (92.4 with JSON+CoN). Llama-3.1-70B on S: 33.4 | 500 (implied, not printed) | GPT-4o, Llama 3.1, Phi-3 | off-J (>97% agreement with humans) | full ~115k tokens | – | MIT | no | V (page image). The widely quoted "60.2" is Zep's own rerun |
| LongMemEval paper, best RAG [LME] | **M** orig | 72.0: GPT-4o reading top-10 rounds, fact-expanded keys, Stella V5 retriever | 500 | GPT-4o with CoN+JSON | off-J | top-10 rounds | fact extraction cost not priced | MIT | caches only | V. This is on M, not S |
| Zep [ZepP] | S orig | 71.2 (4o), 63.8 (4o-mini). Zep's FC rerun: 60.2 / 55.4 | 500 (implied) | 4o / 4o-mini | GPT-4o with LME prompts | top-10 nodes and edges, ≈1.6k tokens | 2.58 s / 3.20 s, vs 28.9 s for FC | Graphiti Apache-2.0 | no | V. Scores 17.7 points below FC on single-session-assistant |
| Emergence: Simple Fast / Simple / Internal [Emer][EmerGH] | S orig (the code asserts this file) | 79.0 / 82.4 / 86.0. Their FC 63.8; their oracle 82.4 | 500 | gpt-4o-2024-08-06 | code uses off-J prompts; blog does not name the judge | 42 turns (20 in "Faster") | median 3.59 / 7.12 / 5.65 s | Simple Fast code only, no licence | no | V. Their oracle (82.4) is far below the paper's 87.0 |
| Mastra RAG [MastraRAG] | S | 80.05 at topK 20 (63.41 at topK 2) | 500 | gpt-4o | not named | topK 20 + last 10 messages | – | Apache-2.0 (except ee/) | no | V |
| Mastra Observational Memory [MastraOM] | S orig per their guide | 94.87 (gpt-5-mini), 84.23 (gpt-4o) | 500 | listed | gpt-4o with LME prompts | ≈30k tokens | observer: gemini-2.5-flash | Apache-2.0 core | no (results/ is gitignored) | C. The "~10% judge swing" remark is about LoCoMo. Repo workflow iterates on failed test questions |
| Supermemory [SM] | S | 81.6 / 84.6 / 85.2 (reader 4o / GPT-5 / Gemini-3) | 500 | listed | gpt-4o with LME prompts | – | – | MIT | no | V. OmniMemEval rerun: 66.07 |
| Hindsight paper [HS] | S (version not stated) | 91.4 (Gemini-3 generates answers only), 89.0 (OSS-120B), 83.6 (OSS-20B) | 500 | listed | **GPT-OSS-120B** | cells read "<add>" | – | MIT | viewer not confirmed | C. Baseline rows are copied from reports that used a GPT-4o judge |
| Hindsight v0.4.19, in the vendor's own harness (AMB) [AMB] | **S cln** | **94.6 (473/500)** | 500. Abs questions graded with category prompts, no abstention prompt | gemini-3.1-pro-preview | gemini-2.5-flash-lite | **43.6k tokens of context** | ingestion 8.4 h; retrieval 675 ms | MIT (AMB repo has no licence) | yes, s.json.gz | A. Vendor harness; answer prompt tuned to LME |
| AMB "hybrid-search" baseline [AMB] | S cln | 74.0 (370/500) | 500 | same | same | top-50 chunks of 512 tokens, 23.2k tokens | ingestion ≈16 h (local embedding) | AMB (no licence) | yes | A. Closest public analogue of dense+sparse RRF |
| Honcho [Honcho] | S | 90.4 (452/500). Gemini 3 Pro alone with FC: **92.0** | 500 | claude-haiku-4-5 | GPT-4o with LME prompt | median 5% of history | ingestion: gemini-2.5-flash-lite | AGPL-3.0 | per-commit folders | V |
| EverMemOS [EMOS] | S | 83.00 | 500 | GPT-4.1-mini | 4o-mini + 2 auxiliary judges | 10 MemScenes / 10 episodes | – | Apache-2.0 | no | C. Baselines copied from the MemOS leaderboard; the "79.6 / 71.2" ablation is not in the paper |
| MemOS, in its own OmniMemEval harness [OMEval] | S cln | 89.20 (100 on all three single-session types) | 500 | gpt-4.1-mini | gpt-4o-mini | 4,151 tokens | – | Apache-2.0 | snapshot only | V. Vendor runs the harness |
| OmniMemEval reruns (by MemTensor, a competitor) [OMEval] | S cln | graphiti-zep 79.80 (at 117k tokens), EverOS 80.40, mem9 78.00, Letta 77.67, Hindsight 72.20, Supermemory 66.07, MemMachine 63.60, Viking 61.07, Mem0 56.00, Cognee 51.80, Memori 20.80 | 500 | gpt-4.1-mini | gpt-4o-mini | varies | – | Apache-2.0 | snapshot | S. Cells such as Hindsight single-session-assistant 14.29 and Mem0 single-session-user 8.57 look like adapter failures |
| TiMem [TiMem] | S | 76.88 ±0.30 (4o-mini), 78.96 (4o) | 500 | listed | official prompts, but the table labels the judge gpt-4o-mini | k=20, 1,271 tokens | P50 1.76 s | Apache-2.0 / MIT | – | V |
| SmartSearch [Smart] | S | 88.4 index-free; its FC baseline 65.6 | 500 (text is ambiguous about abs) | gpt-4.1-mini | gpt-4o-mini | 3,392 tokens | ≈650 ms on CPU | no code | no | C. The +6 pp reranker gain is from MiniLM to **bge**-large |
| JustMem [JustMem] | S | 83.40 ±0.24 | 500 | GPT-4.1-mini | GPT-4.1-mini (judges its own answers) | 1.70k inference tokens | – | no code | no | N |
| JordanMcCann agentmemory [JMcC] | S | 96.2 (481/500) | 500, "zero skips" | Claude Opus 4.6 | gpt-4o with the official templates verbatim | 500 candidates + cross-encoder | ≈$1,000 of development spend | MIT | yes | V. Tuned on the test set for 16 days |
| Chronos [Chronos] | S | 95.6 (Opus 4.6), 92.6 (GPT-4o) | 500 | listed | LME judge; model not named | – | – | none | no | V. Baselines cited, not rerun |
| Agent Zero [AZ] | ? | 95.60 (gpt-5.5); 92.2–95.6 across 8 backbones | 500 | gpt-5.5 | not named | 12.2k prompt tokens | $0.0348 per question | none | no | C. The "+0.73" is measured against Mastra OM |
| OMEGA [OMEGA] | ? | 95.4 (466/500) | 500 | GPT-4.1 | GPT-4.1 (judges its own answers) | – | – | Apache-2.0 | no | self-judged |
| Mem0 platform v3 [Mem0B] | S cln | README: 94.4 / 94.8. **Committed files: 93.4 (467) / 90.4 (452)** | 500 | gpt-5 | gpt-5 with a custom lenient judge | top-200 / top-50, under 7k tokens | latency fields read 0.0 | Apache-2.0 | yes (older runs only) | C. The README change was a README-only commit; prompts contain hints specific to test items |
| Mem0 OSS [Mem0B] | S cln | 91.0 (GPT-5 extraction) | 500 | gpt-5 | gpt-5, lenient | top-200 | ≈125k extraction calls (estimate) | Apache-2.0 | yes | A |
| Backboard [BBlme] | S cln | 93.4 (467/500) | 500 | gpt-4.1 | gpt-4o-mini, **chosen after the fact as the most lenient of 4 judges** | – | – | no licence | no | V |
| ByteRover [BR] | S (cleaned claimed) | 92.8 (464/500) | 500 | Gemini 3.1 Pro | Gemini 3 Flash | – | p50 1.6 s | brv-bench, no licence | – | V |
| Maximem Synap [Maxi] | ? | 92.0 | 500 | gpt-5-mini | gpt-5-mini (judges its own answers) | – | – | harness MIT; engine closed | on request | C. The reruns of other vendors in their harness were not found |
| Memanto [Memanto] | S | 89.8, reached through stages 56.6 → 89.8 | 500 | Gemini 3 | Claude Sonnet 4 | up to 100 chunks | under 90 ms | eval repo has no licence | – | C. Tuned on the test set |
| Engram (ahammadnafiz) [EngA] | S | 90.8 (454/500) | 500 | claude-sonnet-4-6 | claude-opus-4-8; LME judge prompt not documented | 60 memories, 17.7k tokens | $0.074 per question; p50 25.7 s | Apache-2.0 per LICENSE file | not published | C |
| MemMachine [MM] | S | 93.0 (GPT-5-mini, k=100) | 500 | GPT-5 / GPT-5-mini | gpt-4o-mini with LME prompts | episode clusters | – | Apache-2.0 | – | V. The 93.0 comes from a 12-configuration ablation run on the test set |
| Mandol [Mandol] | S cln | 85.00 (4o-mini), 88.40 (4.1-mini) | 500 | listed | gpt-4o-mini via the EverMemOS script | 2.1–2.3k tokens | – | Apache-2.0 | no | C. The +7.2 margin is over MemOS, and the baselines are copied |
| EMem-G [EMem] | S | 77.9 (4o-mini), 84.9 (4.1-mini) | **470** (abs excluded) | listed | gpt-4o-mini, 3 runs | 1.0–3.6k tokens | – | MIT | – | C |
| Fidelity Before Structure [FBS] | S | verbatim chunks 67.4 vs extracted artifacts 45.4 | 500 | gpt-4o | gpt-4o-mini (κ 0.897 with humans) | 5k-token cap | $12.5 per 1,000 correct | MIT | cached outputs | V |
| LightMem [LightMem]; ielab reproduction [ielab] | S | 68.64 (4o-mini). Reproduction: naive RAG 67.3 vs LightMem 68.2–70.7 | 500 (5 corrupt items counted wrong); reproduction uses 444 (single-session-assistant excluded) | 4o-mini / Qwen3-30B | 4o-mini / gpt-5.5 | – | – | MIT | – | V / S |
| Zep 2026 research page [ZepR] | ? | 90.2 (451/500) | 500 | gpt-5.4 | gpt-5.4 with CoT, prompt unpublished | 4,408 tokens | – | – | no | U |

### 1B. LoCoMo: answer metrics

The official metric is Porter-stemmed token F1 over 1,986 questions, including category 5. Nearly all vendors instead report J over the 1,540 questions in categories 1–4.

| System [src] | Subset | Metric | Value | Denominator | Reader | Judge | Budget | Cost/latency | Code (licence) | Per-question artifacts | Verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|
| LoCoMo paper [LoCoMo] | all 5 categories; 10 vs 50 conversations not stated | F1 | Human 87.9; GPT-4-turbo 32.1; gpt-3.5-16K FC 37.8; best RAG 41.4 (top-5 observations) | includes adversarial | listed | none | – | – | data CC BY-NC 4.0 | – | V |
| Mem0 paper [Mem0P] | categories 1–4 | J, 10 runs | Mem0 66.88 ±0.15; Mem0g 68.44; **FC 72.90**; Zep 65.99; best RAG 60.97 | 1,540 | 4o-mini | 4o-mini, Mem0-gen | per-query context: 1,764 / 3,616 / 26,031 tokens | search p50 0.148 s; FC total p50 9.87 s | Apache-2.0 (legacy harness, commit aae5989) | no | C. Labels rotated; the token column is per query, not stored memory |
| Zep, 2025 rebuttal [Zep25] | categories 1–4 | J | 75.14 ±0.17 (grades file: 1157/1540 = 75.13) | 1,540 | 4o-mini | 4o-mini, Mem0-gen | – | search p95 0.632 s | no licence | yes | A. Categories 2 and 3 swapped. The original ~84 had a numerator/denominator bug. Mem0's rerun of Zep: 58.44 |
| **Zep, December 2025** [ZepDec] | categories 1–4 | J, 10 runs | **80.32 ±0.43 at 1,997 tokens**; 80.06 at 1,378; 77.06 at 756; 73.72 at 504; 69.62 at 347 | 1,540 | 4o-mini | 4o-mini, Mem0-gen | edge/node limits | retrieval p50 0.189 s | Apache-2.0 | yes, 10 runs per config | V. The cleanest accuracy-vs-token curve available |
| Zep 2026 page [ZepR] | ? | accuracy | 94.7 (1,459/1,540) | 1,540 | gpt-5.4 | gpt-5.4 CoT | 5,760 tokens | p50 87 ms | – | no | U. Per-category counts don't sum; above the 93.57 ceiling |
| Mem0 v3 [Mem0B] | categories 1–4 | lenient J | README 92.5 / 91.8. **Committed files: 91.56 (1410) / 82.66 (1273)** | 1,540, with 156 / 274 rerun questions merged in | gpt-5 | gpt-5 (dates within 14 days, durations within 50%, evidence may override gold) | top-200 ≈6,956 tokens | search p50 3.1 s | Apache-2.0 | yes | A. Issue #30: same answers score 88.46 with the old judge vs 94.32 with the new |
| Letta [Letta] | categories 1–4 | SimpleQA-style grader | 74.0 | 1,540 expected | gpt-4o-mini agent | gpt-4.1 | tool-using agent | – | no licence | no | V |
| MemOS-1031 [MemOS] | categories 1–4 | J + F1 | 75.80 (F1 45.27). In its own harness: 88.83 | 1,540 | 4o-mini / 4.1-mini | not stated / 4o-mini | 1,589 / 5,400 tokens | – | Apache-2.0 | – | V (the 88.83 is vendor-run) |
| Memobase [Memobase] | categories 1–4 | J | 75.78 (1167/1540) | 1,540 | Mem0 harness default | "e.g. gpt-4o" | – | – | Apache-2.0 | yes | A, rot |
| EverMemOS / EverOS [EMOS][EverOS] | categories 1–4 | J, average of 3 judges | 93.05 (4.1-mini), 86.76 (4o-mini). EverOS README: 94.42 | 1,540 | 4.1-mini | 4o-mini + 2 auxiliary (EverOS: 1 run) | claimed 2.3k; the paper's own Table 8 shows ≈6.7k per question | – | Apache-2.0 | results repo returns 404 | C. Issue #73 got 38.38 on conv-26; LoCoMo-Refined rescores it at 58.25 |
| Backboard [BBlc] | categories 1–4 | J | 90.00 (1386/1540) | 1,540 | gemini-2.5-pro | GPT-4.1, Mem0-gen | – | – | no licence; redistributes the CC BY-NC data | yes | A, rot. OmniMemEval rerun: 22.40 |
| Hindsight paper [HS] | categories 1–4 | J | 89.61 (1380/1540, Gemini-3 answering) / 85.67 / 83.18 | 1,540 | listed | GPT-OSS-120B | – | – | MIT | none found | C, rot. Baselines copied from Backboard |
| Hindsight in AMB [AMB] | categories 1–4 | J | 92.01 (1417/1540) | 1,540 | gemini-3.1-pro-preview | gemini-2.5-flash-lite | **36.2k tokens, more than a whole conversation** | – | MIT | yes | A. Effectively full-context reading |
| AMB hybrid-search [AMB] | categories 1–4 | J | 79.09 (1218/1540) | 1,540 | same | same | 22.2k tokens | 219 ms | – | yes | A |
| MIRIX [MIRIX] | categories 1–4 | J | 85.38; **FC 87.52** | 1,540 | 4.1-mini | GPT-4.1 | – | – | Apache-2.0 | – | V. Labels are in sequential ID order, not the code mapping |
| Nemori [Nemori] | categories 1–4 | J | 80.8 vs FC 80.6 (4.1-mini); 73.0 vs 72.3 (4o-mini) | 1,540 | listed | 4o-mini | 2,745 vs 23,653 tokens | – | MIT | – | V |
| LightMem [LightMem] | ? | J | 72.99 vs FC 71.83; on Qwen, 72.60 is **below** FC 74.87 | ? | 4o-mini / Qwen | 4o-mini | – | 85k tokens total | MIT | – | V |
| MemMachine v0.2 [MM] | categories 1–4 | J | 91.69 (4.1-mini, agent mode). With gpt-5-mini, **the no-memory baseline scores 91.7, above MemMachine's 90.5** | 1,540 | listed | 4o-mini | – | 4.20M input tokens | Apache-2.0 | – | V |
| Memori [Memori] | categories 1–4, deduplicated | J | 81.95 | 1,529 | 4.1-mini | 4.1-mini (judges its own answers) | 1,294 tokens | – | – | – | C. Baselines copied from MIRIX |
| SmartSearch [Smart] | categories 1–4 | J | 91.9 (MemOS protocol) / 93.5 (EverMemOS protocol). FC 77.1 | 1,540 | 4o-mini / 4.1-mini | same as reader | 3,141 tokens | 650 ms on CPU | no code | – | V |
| Continua [Continua] | categories 1–4 | J | 74.4 (4o-mini) → 84.5 (4.1-mini) with the **same context and judge** | 1,540 | listed | 4o-mini, Mem0-gen | – | – | none | no | V (the 84.5 is labelled preliminary) |
| Memanto [Memanto] | categories 1–4 | J | 87.1 | 1,540 | Gemini 3 | Claude Sonnet 4 | – | – | no licence | – | C, rot |
| JustMem [JustMem] | categories 1–4 | J | 79.61 ±0.12 | 1,540 | 4.1-mini | 4.1-mini (judges its own answers) | 1.33k tokens | – | none | – | N |
| Synthius-Mem [Synth] | includes adversarial | J | 94.37 | 1,813 (derivation not explained) | 4.1-mini | 4.1-mini (judges its own answers) | – | – | none | – | C |
| locomo-audit [audit] | categories 1–4 | J | FC with gpt-4.1-mini + CoT prompt: **92.62**. Score ceiling from gold errors: **93.57** | 1,540 | 4.1-mini | 4o-mini, Mem0-gen | ~26k tokens | – | CC BY-NC 4.0 | yes | V |
| LoCoMo-Refined [Refined] | 1,382 revised questions | stricter J | MemoraX 82.65, MemOS 63.60, MemPalace 58.68, EverMemOS 58.25, Mem0 48.91 | 1,382 | per system (not verified) | Qwen3-14B (86.33% agreement with humans vs 43.67% for the original judge) | – | – | CC BY-NC 4.0 | – | V |
| SeCom [SeCom] | official QA (appendix) | GPT4Score, 0–100 | 84.21 (turn-level 81.52, FC 66.28) | ? | GPT-35-Turbo | GPT-4-0125 | ≈3.5k tokens | – | MIT | – | V. Not a J score |
| EMem [EMem] | categories 1–4 | J | 0.780 (4o-mini), 0.842 (4.1-mini) | – | listed | 4o-mini, 3 runs | ≈738 tokens | – | MIT | – | V |
| Mandol [Mandol], memU [memU], CueMem [CueMem] | ? | J | 89.48 / 92.21 on "1,986" questions; 92.09 ("historical"); 84.07 (self-judged) | unclear | – | – | – | – | – | – | U |

### 1C. Retrieval and hit-rate claims (no reader; not comparable with 1A or 1B)

| System [src] | Data | Metric as actually defined | Value | Denominator | Verdict |
|---|---|---|---|---|---|
| LongMemEval paper [LME] | M orig | "Recall@k"; any vs all not stated, probably all | session K=V+fact: R@5 73.2, R@10 86.2. round: 64.4 / 78.4 | 470 | V |
| Engram [EngA] | S | at least one gold session among 60 memories (≈17.7k tokens) | **100** (470/470); mean session recall 99.7 | 470 | V |
| MemPalace [MemPal] | S cln | session recall_any@5 | raw 96.6; hybrid 98.4 on a 450-question held-out split; "100" | 500 including abs; 450 | V (the 100 is contaminated) |
| MemPalace on LoCoMo [MemPal][MemPalH] | 1,986 including adversarial | fractional evidence recall at session level; empty evidence scores 1 | R@10 88.9 (hybrid v5); 60.3 raw; "100" was retracted | 1,986 | C |
| rohitg00 agentmemory [rohit] | S cln | session recall_any | R@5 95.2, R@10 98.6 (BM25 alone: 86.2) | 500 stated | V |
| Supermemory [SM] | S | "Recall@15 / @20", undefined | 95 / 97, shown next to Zep's QA accuracy | – | V (misleading presentation) |
| FluctlightDB [Fluct] | "v4 unified" | session_recall@8 | 97.6 | 500 including abs | V (non-standard) |
| JustMem [JustMem] | S / LoCoMo | fraction of gold evidence within the top 10 selected | 96.00 / 88.05 | 500 / 1,540 | N |
| LiCoMemory [LiCo] | ? | fraction of gold targets in the top 15 | 76.63 | 500 | V |
| Engram v0.1.3, locomo issue #38 [EngI38] | LoCoMo | session recall_any (not turn-level) | R@5 93.9, R@10 95.0 | 1,982 including adversarial | C |
| Lexical-dense fusion [Fusion] | LoCoMo | session Hit@1 / R@5 | fusion 0.752 / 0.894 vs BM25 0.640 / 0.833 | 1,978 | V |
| Entity-memory graph [EMG] | LoCoMo | official fractional evidence recall | at k=25: 84.48 vs dense 79.75; **no F1 difference** | 1,986 | V |
| LazyMem [Lazy] | own 100-question LME split; 314 LoCoMo questions | All@50: complete gold coverage after ±w neighbour turns | LME 0.950 → 0.980; LoCoMo 0.781 → 0.889 | 100 / 314 | V |
| SmartSearch [Smart] | LoCoMo | candidate-pool recall / gold surviving truncation without rerank | 98.6 / 22.5 | 1,540 | V |
| memorybench [memorybench] | any | "recall@k" = an LLM judges whether any result is relevant to the gold answer | – | – | V (code). It is a hit rate |

### 1D. Subsets and non-standard settings (low weight)

- LongMemEval commercial pilot: 97 questions with short histories [LME].
- Honcho on LongMemEval M: 88.8 on 98 questions, selection method not described [Honcho].
- Other small or custom subsets: LazyMem uses 100 LME questions; ReFind reports 93.2 on 50 questions [ReFind]; MEMTIER uses 100 [MEMTIER]; PACMS uses 100 [PACMS]; Retain-or-Consolidate tests on 75 [RoC]; the fusion paper uses 150 LME-S questions; Chronos runs ablations on 116.
- RMM reports 70.4 without specifying the variant [RMM].
- MemoryAgentBench's LME(S*): 5 contexts of ~355k tokens and 300 questions [MAB].
- LongMemEval-V2: 451 questions; AgentRunbook-C reaches 72.5 or 74.9 depending on the tier; reader fixed to Qwen3.5-9B [LMEv2].
- BEAM uses nugget scoring [BEAM]. LoCoMo-Plus adds a 401-question "Cognitive" split [LoCoPlus].

---

## 2. What the "100%" claims actually measure

- **Engram 100%** [EngA]: a retrieval hit rate. It counts a question as a hit if at least one gold session appears among 60 retrieved memories (about 17.7k tokens, roughly 17% of the haystack), over the 470 answerable questions. No LLM is involved. Its answer accuracy is a separate number: 90.8% on 500, with a Claude Sonnet reader and a Claude Opus judge whose LME prompt is undocumented.
- **MemPalace LongMemEval 100%** [MemPal]: session recall_any@5 with a Haiku reranker, after three fixes targeted at the three test questions it failed. The authors call this "teaching to the test". The held-out figure is 98.4% on 450 questions. The raw 96.6% is simply MiniLM session retrieval, over 500 questions including the 30 abs that the official retrieval eval skips.
- **MemPalace LoCoMo 100%** [MemPalH]: fractional evidence recall with top_k=50. Every conversation has only 19–32 sessions, so this returns everything. It was retracted on 2026-04-14. The honest figures are 60.3 raw and 88.9 hybrid (R@10 over 1,986 questions).
- **Backboard 99.95%** [BB99]: the model was LoRA-trained on the very conversations it was tested on. This is memorisation. By my inference it is 1985/1986.
- **Supermemory "~99%"** [SM]: self-described as a parody. It reportedly counted a question correct if any of 8 prompt variants got it right (secondary source).
- **Per-category 100s** (MemOS on the three single-session types, Maximem, Emergence single-session-assistant, Mastra single-session-assistant, agentmemory abstention 30/30): these are small, easy buckets (single-session-assistant has n=56, preference n=30) and are often self-judged or run in the vendor's own harness. Chronos flags gold label 6d550036 as questionable, so even these buckets have label noise.
- **Retrieval-hit claims in the high 90s** (MiniLM + BM25 at 95.2 R@5): this shows session-level recall_any on LME-S is nearly saturated by cheap retrievers. It says nothing about recall_all, turn-level evidence recall, or answers.

---

## 3. Benchmark and judge flaws that affect comparisons

**LongMemEval**
1. **Denominators.** QA is scored on 500, including abs. Official retrieval scoring skips the 30 abs and any question with no user-side `has_answer` turn; that second count is printed at runtime and not published. The official aggregates are overall accuracy, task-averaged accuracy and abstention accuracy, and results should say which one is reported.
2. **Retrieval metrics.** The official code indexes user turns only. It logs recall_any@k, recall_all@k and ndcg_any@k for k in {1,3,5,10,30,50}. The paper just says "Recall@k", which is probably recall_all. Vendors usually report recall_any@5.
3. **Versions.** The cleaned release changes 21 rows covering 20 question IDs: 17 sessions removed, 4 no-ops. The oracle file is byte-identical across releases. The HF `answer` column mixes int and str. Most vendors don't say which version they used.
4. **Anchors quoted wrongly.** Vendors cite "FC 60.2" (Zep's rerun) and "oracle 82.4" (Emergence's run). The paper's own figures are 60.6 and 87.0–92.4.
5. **Judges vary.** Setups range from off-J to GPT-OSS-120B, gemini-flash-lite, gpt-4o-mini, Claude judges, and models grading their own answers. Backboard picked the most lenient of four judges. AMB and memorybench grade abs questions with non-abstention prompts. Mem0's harness judges with "lean toward yes" and its prompts carry hints tied to specific test items. Official judge agreement with humans is weakest on preference (0.90) and abstention (0.90 when judging Llama-8B answers).
6. **The reader dominates.** Gemini 3 Pro with full context scores 92.0. Haiku 4.5 goes from 62.6 with full context to 89.2 with oracle evidence. o3 goes from 76.0 to 92.0. A score of 90–96 without a same-reader FC and oracle baseline cannot be interpreted.
7. **Development on the test set.** Seen in JordanMcCann (16 days), Mastra (investigation workflow), MemMachine (12-configuration ablation), Memanto (staged tuning), MemPalace, and Chronos (no split disclosed).

**LoCoMo**
1. **Metric.** The official metric is stemmed F1 over 1,986 questions, with a keyword check for category 5. Evidence recall is fractional over dia_ids, and empty evidence counts as 1. Vendors report J on 1,540 questions instead.
2. **Category mislabelling.** Mem0, Memobase, Backboard, Hindsight and Memanto use the rotated labels. Zep 2025 swaps categories 2 and 3. MIRIX uses sequential labels. supermemory's memorybench maps 1=single and 2=multi. Overall scores are unaffected, but per-category comparisons across papers are unreliable.
3. **Gold quality.** 99 of 1,540 gold answers (6.4%) are wrong, giving a ceiling of 93.57. There are also 57 citation-only errors, 12 duplicated questions, and a category-5 formatter bug (the code reads `answer`, but adversarial rows use `adversarial_answer`).
4. **Judge leniency.** gpt-4o-mini with Mem0-gen accepted 62.81% of deliberately vague wrong answers. The mem0 v3 judge accepts dates within 14 days and durations within 50%. The same answers score 88.46 or 94.32 depending on the judge. A stricter Qwen3-14B judge drops systems by 15–22 points.
5. **Full context fits.** Conversations are ≈16.6–26k tokens. FC scores 72.9 (4o-mini), 80.6 (Nemori's setup, 4.1-mini), 87.52 (MIRIX's setup, 4.1-mini), and 92.62 (4.1-mini with CoT). A gpt-5-mini reader with no memory at all scores 91.7. Any system using more context than about a fifth of the conversation tests reading, not memory. AMB fed 22–36k tokens.
6. **Statistics.** Swapping only the reader moves scores by about 10 points (Continua). With Wilson CIs, 56% of adjacent leaderboard comparisons are indistinguishable. Open-domain (n=96) needs a gap of 15+ points to separate two systems.
7. **Artifacts vs headlines.** Mem0's README numbers are not backed by committed files. Zep's original ~84 counted category-5 correct answers in the numerator but not the denominator. The Zep 2026 counts are internally inconsistent.
8. **Licence.** The data is CC BY-NC 4.0, which restricts commercial reuse.

---

## 4. Reproduction shortlist

All of these run on remote APIs only, which suits a weak laptop.

**Price assumptions (per 1M input/output tokens, OpenAI list prices from memory, not re-fetched today):** gpt-4o-2024-08-06 $2.50 / $10; gpt-4o-mini $0.15 / $0.60; gpt-4.1-mini $0.40 / $1.60. Re-verify before binding a priced plan (AGENTS rule 5).

**Token sizes:** LME-S is ≈57.5M haystack tokens across 500 questions. The oracle file is ≈15.4 MB, about 7–8k tokens per question (my estimate). LoCoMo has 10 conversations of ≈16.6–26k tokens each.

**HybridMind constraints:** reproductions that use MiniLM or Stella must stay external baseline scripts. The HybridMind arm must use the native remote 4096-d embedder (rule 3); Qwen3-Embedding-8B is one option, and it is what LazyMem uses. Every run must go through the preflight plan gate.

**Always-on baselines, so every number can be interpreted:**
- Same-reader full context: LME-S ≈57.5M tokens, about $8.6 with 4o-mini; LoCoMo ≈40M tokens, about $6 with 4o-mini, less with prompt caching.
- Same-reader oracle: LME ≈3.8M tokens, about $0.6 with 4o-mini.
- off-J on 500: ≈0.3M input tokens, about $1.

### R1. LongMemEval official RAG pipeline (faithful; MIT) [LMEgh]

- **Unit:** value = round (a user and assistant turn pair). Key = the user utterances plus extracted user facts, **merged into one key** (key merge, not rank merge).
- **Retriever:** `flat-stella` (Stella V5 1.5B) as in the paper; `flat-bm25` or `flat-contriever` also available.
- **Retrieval:** top-10; also report top-5.
- **Context order:** items sorted by timestamp.
- **Reader:** method `con` (Chain-of-Note) with JSON history format. Use `gpt-4o-2024-08-06` for a faithful run, and gpt-4o-mini for the budget arm, which has no published anchor on S.
- **Judge:** `evaluate_qa.py` off-J; `print_qa_metrics.py` asserts the judge model.
- **Retrieval scoring:** `eval_utils` recall_any and recall_all at session and turn level, with the exclusion count logged.
- **Anchors:**
  - M (Table 3): R@10 0.784; GPT-4o QA 0.720 (top-10) and 0.657 (top-5).
  - S (Table 8, round values, K=V+fact): e.g. Mistral-Nemo 0.666, Llama-3.2-3B 0.508.
  - S bounds: FC 0.606, oracle 0.870 / 0.924.
- **Cost on LME-S:**
  - Reader: ≈3–6k input + ≈0.4k CoN output per question. About $6–10 with gpt-4o, about $0.5 with 4o-mini.
  - Judge: about $1.
  - Fact expansion: reuse the released key-expansion caches. They were built on the original data, but the cleaned release only removes sessions, so reuse is probably valid; this is my inference and should be checked. Otherwise, roughly $5–15 with a mini-class model; this is a loose estimate that depends on dedup of shared filler sessions (also unverified).
  - Embedding: at most ≈57.5M tokens before dedup.
- **LoCoMo:** not in the paper. Apply the same pipeline with LoCoMo turns as values and score with dia_id evidence recall. Cost is under $1 with 4o-mini.

### R2. Emergence "Simple Fast" to "Simple" (clear mechanism; reimplement, because the repo has no licence) [EmerGH][Emer]

- **Unit:** every turn, user and assistant, stored as `"[session date] role: content"`.
- **Retrieval:** cosine top-42 turns (MiniLM in the original). "Faster" uses top-20.
- **Reader, two calls at temperature 0:**
  1. CoT extraction of the relevant facts from the 42 turns (max 512 output tokens).
  2. Answer from the facts, the raw turns and `question_date` (max 256 output tokens).
  - Pin the reader to `gpt-4o-2024-08-06`; the code uses the unpinned alias `gpt-4o`.
- **"Simple" (82.4):** add a cross-encoder rerank of the turns, group turns into sessions, score each session by the NDCG of its turns, and pass whole sessions. The number of sessions passed is not published.
- **Judge:** the official prompts are already copied into the code, including abstention.
- **Data:** the code asserts the original file. Run both original and cleaned.
- **Anchors:** 79.0 (Simple Fast), 76.8 (Faster), 82.4 (Simple). Their own FC: 63.8.
- **No ingestion LLM calls.**
- **Cost on LME-S:** ≈21k input and ≈0.7k output tokens per question (estimate). About $30 with gpt-4o, $5 with gpt-4.1-mini, $1.8 with 4o-mini, plus about $1 for the judge.
- **Cost on LoCoMo:** turns average ≈28 tokens, so 42 turns ≈1.2k tokens and ≈4k per question including prompts. About $1 with 4o-mini.

### R3. LoCoMo under the Mem0-paper protocol, with Zep's December 2025 budget curve as the target [Mem0legacy][ZepDec][Mem0P]

- **Faithful baselines:** FC (26,031 tokens) at 72.90 ±0.19, and the RAG chunk-size × k grid (best: k=2 with 256-token chunks, 60.97). Reader and judge are gpt-4o-mini at temperature 0 with Mem0-gen and 10 judge runs. The harness is Apache-2.0 at commit `aae5989`; it was retired from the repo but remains in git history.
- **Target frontier:** Zep December 2025, with released per-run files: 69.62 at 347 tokens → 73.72 at 504 → 77.06 at 756 → 80.06 at 1,378 → 80.32 at 1,997, over 10 runs. Compare any fixed-budget selector at matched median context tokens.
- **Report alongside:**
  - official stemmed F1 over all 1,986 questions and dia_id evidence recall (flag the empty-evidence = 1 rule);
  - an all-evidence ("recall_all") variant;
  - accuracy on the 1,441 questions not flagged by the audit;
  - per-category results under the code mapping, with Wilson CIs.
- **Cost:** reader 1,540 × 2–4k tokens ≈ $0.5–1 with 4o-mini; judge ×10 runs ≈ $0.7; FC ≈ $6; embedding ≈0.26M tokens.
- **Optional:** the AMB hybrid-search retrieval config (Qwen3-Embedding-0.6B + BM42, RRF, 512-token chunks, top-50) has public per-question files [AMB]. Its gemini-3.1-pro reader costs about $23 on LME and about $70 on LoCoMo (estimate), so compare retrieved contexts only.

### Not shortlisted

- **Hindsight:** MIT with artifacts, but ingestion took 8.4 h with tens of thousands of LLM calls, it reads 43.6k tokens of context, runs in a vendor harness, and scores 72.2 in a competitor's harness.
- **Mem0 OSS:** extraction alone is about $40–75+.
- **Mastra OM:** observer cost about $80–120, no artifacts, and test-set iteration in the repo.
- **EverMemOS / MemOS / Graphiti with Neo4j:** too heavy for a 16 GB machine; disputed reproductions.
- **SmartSearch, Chronos, Agent Zero:** no code.

---

## 5. Mechanism evidence

| Goal | Mechanism | Measured effect (setting) | Status |
|---|---|---|---|
| Recover missing evidence | Fact-augmented keys, merged into the key | LME-M rounds: R@10 0.692 → 0.784; GPT-4o QA 0.670 → 0.720. Average +9.4 recall, +5.4 accuracy. Replacing the key hurts: facts-only R@10 0.654; summary-only session QA 0.252 [LME] | **Measured win** (primary, image-checked) |
| Recover missing evidence | Key merge vs post-hoc rank fusion of expansion keys | R@10 0.784 vs 0.568; QA 0.720 vs 0.596 [LME Table 10] | **Measured**: fuse at index time |
| Recover missing evidence | BM25 + dense fusion | LoCoMo session Hit@1 0.640 → 0.752 with z-score fusion (RRF 0.718); multi-hop +17.5, temporal +16.2. LME-S (n=150): not significant. MEMTIER (n=100): +0.03 with a knowledge-update regression [Fusion][MEMTIER] | Retrieval win on LoCoMo; **no QA evidence** |
| Recover missing evidence | Better retriever | Moves LightMem from 58.1 (BM25) to 75.5 (Qwen3-Emb-4B) [ielab]. Round-level R@10: Stella 0.784 vs BM25 0.538 [LME] | Measured |
| Recover missing evidence | Time-aware query expansion or filtering | Temporal subset R@10 +11.3 (round) / +6.8 (session) with a GPT-4o extractor; a Llama-8B extractor hurts; no QA reported [LME]. TSM without its temporal module: −6.0 on temporal [TSM]. SwiftMem: recall flat, only latency improves [Swift] | Depends on the extractor; QA effect unproven |
| Recover missing evidence | Graph / entity expansion | +4.7 evidence recall at k=25 on LoCoMo with no change in F1 [EMG]. EMem removing graph/PPR: −1.9 on LME, ~0 on LoCoMo [EMem] | Recall win; **answer win unproven** |
| Recover missing evidence | Query rewriting / HyDE | No controlled LME or LoCoMo result. PPRO: −1.9 F1 when removed (U). SmartSearch PRF: +9.2 pp on long histories | **Unproven** |
| Recover missing evidence | Larger k | Mastra topK 2 → 20: 63.4 → 80.05. MemMachine k 20 → 30: +4.2, then k=50: −2.2. Memanto +20.4 (confounded with a threshold change) | Measured; non-monotonic |
| Keep all multi-hop evidence under a budget | ±w neighbour-turn expansion | All@50: LME 0.950 → 0.980, LoCoMo 0.781 → 0.889; accuracy +0.08 / +0.05; saturates at w=2 (n=100 / 314) [Lazy] | Measured, small n; **no token-matched comparison against larger k** |
| Keep all multi-hop evidence under a budget | Turn hits expanded to whole sessions | Emergence Simple 82.4 vs Simple Fast 79.0, confounded with the cross-encoder | Not isolated |
| Keep all multi-hop evidence under a budget | Set-wise selection (SetR) | Evidence coverage 19.3 → 36.5%; MuSiQue F1 12.50 → 15.43 using ~2.9 passages instead of 5; lower MRR [SetR] | Measured, **not conversational** |
| Keep all multi-hop evidence under a budget | Submodular packing | Beats top-k, MMR and LLMLingua-2 over 3 multi-hop datasets, 4 budgets and 4 readers (about +5.1 F1 at 160 tokens, from a snippet) [RINE]. PACMS on 100 LME questions: evidence recall ties MMR, QA higher (from a snippet) [PACMS] | Measured outside conversation; conversational evidence is tiny and not fully verified |
| Keep all multi-hop evidence under a budget | Cross-encoder rerank | Helps a high-recall pool that is badly ordered: gold rank 195 → 8; bge-large CE +6.0 pp [Smart]. Hurts on an already fused shortlist: Hit@1 −6.88 pp, CI [−9.34, −4.34] [Fusion]. Destroys bridge hops: 78.8 → 9.1 on n=66 [CalFus]. Degrades as K grows [Drown] | **Conditional** |
| Keep all multi-hop evidence under a budget | Rerank only within a fixed set | R@10 unchanged by construction; MRR 0.582 → 0.656 [ConvMem] | Safe pattern; retrieval-only evidence |
| Keep all multi-hop evidence under a budget | Budget size and ordering | Mem0 RAG grid is non-monotonic in chunk size and k. Llama-8B degrades beyond ~3k retrieved tokens [LME]. Chronological order beats relevance order: 44.43 vs 38.40 F1 [OPRAG]. Mid-context burial: 75.8 → 53.8 [LitM] | Measured (mostly non-conversational) |
| Keep all multi-hop evidence under a budget | Unit fidelity | Verbatim chunks beat extracted artifacts by 15.9 (LoCoMo) and 22.0 (LME-S) [FBS]. Rounds beat sessions at equal tokens for GPT-4o (qualitative) [LME]. Consolidation wins only when raw evidence does not fit: at 32 tokens 52.0 vs 4.0; at 256 tokens retention leads by 8.0, CI crosses 0 (75-question test) [RoC] | Measured |
| Turn evidence into supported answers | Chain-of-Note + JSON reading format | Oracle accuracy: GPT-4o 0.862 → 0.924; Llama-70B 0.762 → 0.848 [LME] | **Measured win** |
| Turn evidence into supported answers | Two-step "extract, then answer" CoT | EMem: removing QA CoT drops 77.9 → 73.4. Audit: FC with CoT prompt 92.62 vs the memos prompt 81.95 | Measured |
| Turn evidence into supported answers | Reader strength | +10.1 from swapping the reader alone [Continua]. FC scores 92.0 (Gemini 3 Pro) and 91.7 (gpt-5-mini) | Measured, and the dominant factor |
| Turn evidence into supported answers | Diagnostics | Correct retrieval but wrong answer: 13.2% of questions for GPT-4o, 24.6% for Llama-8B [LME]. Whether the answer survives packing adds ΔR² 0.17–0.27 beyond recall [RINE]. Evidence retrieved is often not used [UtilMem] | Diagnostic |
| Turn evidence into supported answers | Sufficiency-gated abstention | Share of correct answers +2–10% (abstract only, non-conversational) [SuffCtx]. Verbatim chunks abstain worse than artifacts: LoCoMo category 5, 6.5 vs 15.0 [FBS] | Unproven for conversational memory |

---

## 6. Novelty check: "fixed-budget evidence-set selection for multi-hop conversational memory"

**Already published, so the core idea is not new:**
- **PACMS** (2606.20047) [PACMS]: budget-aware submodular (facility-location) selection over one pooled set of memories, turns and tool outputs. Evaluated on a 100-question LongMemEval sample: evidence-round recall ties MMR, QA accuracy is higher with two GPT-5-family readers. The token budgets, whether recall means any or all, the judge, and the result numbers were not verified; the PDF did not parse. Licence CC BY-NC-SA.
- **Recall Is Not Enough** (2607.00725) [RINE]: submodular packing under 4 token budgets, plus the answer-in-context diagnostic. Multi-hop QA only, not conversational; under review at EACL 2027.
- **MEMO** (2609.07471) [MEMO]: selects "necessary evidence under a given budget" across text and visual modalities; 128-token budget; reports LoCoMo F1 with the reader also acting as judge. It is a work in progress, and the extracted numbers look inconsistent (N/U).
- **Retain or Consolidate?** (2607.17545) [RoC]: a budgeted packing policy choosing between retaining raw evidence and generative consolidation, at 16–256 tokens, on LME and LoCoMo samples, with a DeepSeek-V3.2 reader. It cites Kang et al. (arXiv 2606.10616), which learns which notes to retain under constraints (not verified).
- **Conversational neighbours:** LazyMem (complete-evidence coverage All@50 plus neighbour expansion; 213-token contexts) [Lazy]; JustMem (adaptive discovery breadth vs reading fidelity; recall is the fraction of gold evidence in the top 10) [JustMem]; SmartSearch (fixed word budget; measures gold surviving truncation) [Smart]; ConvMemory v2 [ConvMem]; Zep December 2025 budget sweep [ZepDec].
- **Non-conversational neighbours:** SetR [SetR], SEAL-RAG [SEAL], Adaptive-k [AdaK].
- **Snippet-level only, unverified:** MESA (2608.10108; LoCoMo with a 1,000-token budget and k=5) [MESA]; HiGMem (precision and recall of the final evidence set) [HiGMem]; WhenLoss (B=2K budget on LoCoMo) [WhenLoss]; xMemory [xMemory].

**Apparently still open,** as far as these searches found. The search was not exhaustive, and PACMS details are unverified. The gap is less a new mechanism than a controlled evaluation that nobody has published:
1. A selection objective that does not use gold labels and explicitly targets **recall_all** (keeping every supporting turn for multi-hop and multi-session questions) under a hard token budget.
2. Evaluation on the **full** LME-S (500, with both original and cleaned files stated) and LoCoMo (1,540 plus the official F1 over 1,986), using off-J and the Mem0-gen judge respectively.
3. **Token-matched** controls against top-k, MMR, ±w neighbour expansion and simply raising k.
4. Reporting recall_all, answer-in-context and answer accuracy, with same-reader FC and oracle bounds.
5. Per-question artifacts.

On mechanism alone, a novelty claim would be weak. The defensible contribution is the measurement.

---

## Sources

**LongMemEval and related:**
- [LME] https://arxiv.org/abs/2410.10813
- [LMEgh] https://github.com/xiaowu0162/LongMemEval
- [LMEcln] https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned
- [LMEchg] https://docs.google.com/spreadsheets/d/16cHPu2B4XhgC-VvolIoWNs8wwm0Zkbpgu8H9x-qhxWg/edit?usp=sharing
- [LMEv2] https://github.com/xiaowu0162/LongMemEval-V2

**Systems:**
- [ZepP] https://arxiv.org/abs/2501.13956
- [Emer] https://www.emergence.ai/blog/sota-on-longmemeval-with-rag
- [EmerGH] https://github.com/EmergenceAI/emergence_simple_fast
- [MastraRAG] https://mastra.ai/research/use-rag-for-agent-memory
- [MastraOM] https://mastra.ai/research/observational-memory
- [SM] https://supermemory.ai/research/longmembench/
- [HS] https://arxiv.org/abs/2512.12818
- [HSB] https://benchmarks.hindsight.vectorize.io/ and https://github.com/vectorize-io/hindsight-benchmarks
- [AMB] https://github.com/vectorize-io/agent-memory-benchmark (outputs/longmemeval, outputs/locomo, src/memory_bench/memory/hybrid_search.py)
- [Honcho] https://plasticlabs.ai/blog/research/Benchmarking-Honcho
- [EMOS] https://arxiv.org/abs/2601.02163
- [EverOS] https://github.com/EverMind-AI/EverOS/blob/main/benchmarks/README.md
- [OMEval] https://github.com/MemTensor/OmniMemEval/blob/main/docs/user_memory/results.md
- [TiMem] https://arxiv.org/abs/2601.02845
- [LiCo] https://arxiv.org/abs/2511.01448
- [EMem] https://arxiv.org/abs/2511.17208
- [Smart] https://arxiv.org/abs/2603.15599
- [FBS] https://arxiv.org/abs/2601.00821
- [JustMem] https://arxiv.org/abs/2609.19877
- [JMcC] https://github.com/JordanMcCann/agentmemory
- [Chronos] https://arxiv.org/abs/2603.16862
- [AZ] https://arxiv.org/abs/2608.29606
- [OMEGA] https://omegamax.co/benchmarks
- [Mem0B] https://github.com/mem0ai/memory-benchmarks (issue: https://github.com/mem0ai/memory-benchmarks/issues/30)
- [BBlme] https://github.com/Backboard-io/Backboard-longmemEval-results
- [BR] https://www.byterover.dev/blog/benchmark_ai_agent_memory_real_production_byterover_top_market_accuracy_longmemeval
- [Maxi] https://github.com/maximem-ai/memory_and_context_eval_harness
- [Memanto] https://arxiv.org/abs/2604.22085
- [EngA] https://ahammadnafiz.github.io/engram/benchmarks/
- [MM] https://arxiv.org/abs/2604.04853
- [Mandol] https://arxiv.org/abs/2606.29778
- [LightMem] https://arxiv.org/abs/2510.18866
- [ielab] https://arxiv.org/abs/2607.29104
- [ZepR] https://www.getzep.com/research/
- [EngW] https://github.com/ly-wang19/engram
- [Fluct] https://arxiv.org/abs/2608.12365
- [RMM] https://arxiv.org/abs/2503.08026
- [Lazy] https://arxiv.org/abs/2607.22690
- [ReFind] https://arxiv.org/abs/2608.12888
- [MemPal] https://github.com/MemPalace/mempalace/blob/main/benchmarks/BENCHMARKS.md
- [MemPalH] https://github.com/MemPalace/mempalace/blob/develop/docs/HISTORY.md
- [rohit] https://github.com/rohitg00/agentmemory/blob/main/benchmark/LONGMEMEVAL.md

**LoCoMo:**
- [LoCoMo] https://arxiv.org/abs/2402.17753
- [LoCoMoGH] https://github.com/snap-research/locomo (task_eval/evaluation.py; issue #38: https://github.com/snap-research/locomo/issues/38)
- [audit] https://github.com/dial481/locomo-audit and https://penfieldlabs.substack.com/p/we-audited-locomo-64-of-the-answer
- [Mem0P] https://arxiv.org/abs/2504.19413
- [Mem0legacy] https://github.com/mem0ai/mem0/blob/aae5989e78a6188b3b047c104d960c9ad0927e75/evaluation/metrics/llm_judge.py
- [Zep25] https://github.com/getzep/zep-papers/tree/main/kg_architecture_agent_memory/locomo_eval, https://github.com/getzep/zep-papers/issues/5, https://blog.getzep.com/lies-damn-lies-statistics-is-mem0-really-sota-in-agent-memory/
- [ZepDec] https://github.com/getzep/zep/tree/main/benchmarks/locomo
- [Letta] https://www.letta.com/blog/benchmarking-ai-agent-memory/ and https://github.com/letta-ai/letta-leaderboard/blob/main/leaderboard/locomo/locomo_benchmark.py
- [MemOS] https://arxiv.org/abs/2507.03724
- [Memobase] https://github.com/memodb-io/memobase/blob/main/docs/experiments/locomo-benchmark/README.md
- [BBlc] https://github.com/Backboard-io/Backboard-Locomo-Benchmark
- [BB99] https://dev.to/jon_at_backboardio/we-hit-9995-on-the-locomo-memory-benchmark-heres-the-catch-and-why-it-still-matters-3and
- [MIRIX] https://arxiv.org/abs/2507.07957
- [Nemori] https://arxiv.org/abs/2508.03341
- [Memori] https://arxiv.org/abs/2603.19935
- [Continua] https://blog.continua.ai/p/the-locomo-fair-fight
- [Refined] https://github.com/mem-eval-suite/LoCoMo_refined
- [Synth] https://arxiv.org/abs/2604.11563
- [MemR3] https://arxiv.org/abs/2512.20237
- [memU] https://memu.pro/benchmark
- [Cognee] https://www.cognee.ai/ai-memory-benchmarks
- [SeCom] https://arxiv.org/abs/2502.05589
- [AMem] https://arxiv.org/abs/2502.12110
- [CueMem] https://arxiv.org/abs/2609.12354

**Harnesses:**
- [memorybench] https://github.com/supermemoryai/memorybench/blob/main/src/orchestrator/phases/retrieval-eval.ts
- [MAB] https://github.com/HUST-AI-HYZ/MemoryAgentBench
- [BEAM] https://github.com/mohammadtavakoli78/BEAM
- [LoCoPlus] https://github.com/xjtuleeyf/Locomo-Plus

**Mechanisms and novelty:**
- [Fusion] https://arxiv.org/abs/2606.04194 and https://github.com/Chrislysen/opsem
- [EMG] https://arxiv.org/abs/2608.27925
- [ConvMem] https://arxiv.org/abs/2606.10842
- [SetR] https://arxiv.org/abs/2507.06838 and https://github.com/LGAI-Research/SetR
- [RINE] https://arxiv.org/abs/2607.00725
- [SEAL] https://arxiv.org/abs/2512.10787
- [AdaK] https://arxiv.org/abs/2506.08479
- [OPRAG] https://arxiv.org/abs/2409.01666
- [LitM] https://arxiv.org/abs/2307.03172
- [Jin] https://arxiv.org/abs/2410.05983
- [Xu] https://arxiv.org/abs/2310.03025
- [SuffCtx] https://arxiv.org/abs/2411.06037
- [CalFus] https://arxiv.org/abs/2603.28886
- [Drown] https://arxiv.org/abs/2411.11767
- [TSM] https://arxiv.org/abs/2601.07468
- [Swift] https://arxiv.org/abs/2601.08160
- [TReMu] https://arxiv.org/abs/2502.01630
- [MEMTIER] https://arxiv.org/abs/2605.03675
- [PPRO] https://arxiv.org/abs/2607.00017
- [DenseX] https://arxiv.org/abs/2312.06648
- [BestPr] https://arxiv.org/abs/2407.01219
- [MemRer] https://arxiv.org/abs/2605.06132
- [Nano] https://arxiv.org/abs/2604.11628
- [PACMS] https://arxiv.org/abs/2606.20047
- [RoC] https://arxiv.org/abs/2607.17545
- [Kang] https://arxiv.org/abs/2606.10616
- [MEMO] https://arxiv.org/abs/2609.07471
- [MESA] https://arxiv.org/abs/2608.10108
- [UtilMem] https://arxiv.org/abs/2608.30508 and https://github.com/peijunallin/UtilMem
- [HiGMem] https://arxiv.org/abs/2604.18349
- [WhenLoss] https://arxiv.org/abs/2605.24579
- [xMemory] https://arxiv.org/abs/2602.02007