# HybridMind Application Demos — Technical Specification

## Purpose

Create six small, user-facing applications that demonstrate HybridMind as a persistent, inspectable memory layer. Each application is a thin client of the existing FastAPI service and its isolated .mind persistence package; no demo owns a parallel vector store, graph, or retrieval stack.

The suite is both a product showcase and a research harness. It makes semantic recall, exact lexical recall, temporal context, and explicit relationships visible, instead of presenting a chat answer as ungrounded memory.

## Hard platform invariant

All demos run only with HYBRIDMIND_EMBEDDING_DIMENSION=4096.

- The configured remote TEI deployment of Qwen3-Embedding-8B produces every text vector. A returned vector must be exactly 4096 dimensions.
- If TEI is unavailable, unhealthy, or returns another dimension, startup, ingestion, and search fail visibly. There is no local model fallback, dimensional projection, padding, mixed-dimension index, or quality tier.
- Before a guided demo or measurement run, run python scripts/preflight.py. The readiness screen reports the service state and confirmed vector dimension.
- Z.AI / glm-4.6 is the canonical production hosted route. RunPod vLLM is the self-hosted route. For cost-bounded research only, `ai.hackclub.com` may be selected with explicit `HYBRIDMIND_ALLOW_RESEARCH_PROXY=true`; research mode never silently falls through to paid Z.AI.

## Portfolio and user journeys

All demos use one legible pattern: ingest scoped source material; show timeline and relationship context; ask a question; reveal ranked evidence; generate a response supported by that evidence. The response panel always links its source nodes.

| Demo | Primary user | Guided journey | What it proves |
| --- | --- | --- | --- |
| Persistent personal assistant | Individual | Add preferences, meetings, notes, and follow-ups over simulated days; ask “what did I decide about the trip budget and why?”; inspect linked decision turns. | Session and graph links preserve continuity beyond one chat. |
| Support / CRM copilot | Support agent or account manager | Import a sample account, tickets, calls, and product events; ask for a handoff brief or promised next step; open supporting interactions. | Account-scoped memory joins semantic, exact-ID, and relationship retrieval. |
| Professional knowledge desk | Consultant, analyst, or knowledge worker | Add project documents, meetings, and research notes; ask “which decision affects the vendor risk noted last week?”; show the evidence path. | Hybrid retrieval and decomposition support connected work. |
| Tutor memory | Learner and tutor | Record answers, misconceptions, goals, and explanations; request a next lesson; show prior errors and concepts used to tailor it. | Learning history is useful when retained as connected evidence. |
| Project memory | Product or engineering team | Ingest issue notes, decisions, releases, and retrospective items; ask why a trade-off was made and what remains blocked; follow owners and affected work. | A durable graph converts scattered decisions into accountable context. |
| Multimodal field notebook | Researcher, operator, or creator | Add photos with captions and metadata alongside notes; search for a damaged component and inspect the image with related observations. | Visual retrieval can complement the text, graph, and metadata model. |

### Shared script

1. Start with an empty named workspace and mark the source corpus as synthetic.
2. Ingest 15–50 curated items with workspace_id, isolation key, sessionId, source_type, timestamp, provenance, and sensitivity metadata.
3. Show created nodes plus explicit and automatic relationships.
4. Ask one semantic, one exact-keyword, and one multi-hop question.
5. Show retrieved evidence, relationship context, response, and feedback controls: useful, incorrect, stale, and delete.
6. Demonstrate deletion and snapshot/reset while keeping workspaces isolated.

## Goals

- Establish a credible, local-native vector + graph memory demonstration.
- Make provenance first-class: source text, metadata, relationship context, and node IDs are visible behind every response.
- Reuse a consistent demo shell; vary only corpus, labels, schema, and guided questions.
- Measure retrieval quality, latency, and user trust—not just chat polish.
- Keep demos safe to reset through isolated .mind workspaces containing synthetic or consented data only.

## Explicit non-goals

- These are not complete CRM, tutoring, document-management, or collaboration products. They omit billing, full workflow systems, identity management, and broad integrations.
- The assistant may suggest a reply, lesson, or next step; it must not email customers, schedule events, modify external records, or deploy code.
- Do not make medical, legal, financial, educational-outcome, or compliance claims. Professional samples are fictional.
- No 1024-dimensional mode, local fallback, projection, zero-padding, second index, or mixed-dimension migration. 4096 is mandatory.
- ColBERT, GNN, GAE, graph-conditioned embeddings, and learned fusion remain opt-in research features, not baseline demo dependencies.
- Initial multimodal scope is image-plus-caption retrieval only; it excludes video, audio, OCR, facial recognition, and surveillance.

## Architecture and API integration

    Demo UI / guided script
            | HTTPS
            v
    HybridMind FastAPI
            |-- nodes, edges, bulk/session ingest
            |-- hybrid retrieval, graph path, diagnostics
            v
    4096-dim TEI + HybridMind engine
            v
    Per-workspace .mind package
      SQLite WAL + FAISS HNSW + bm25s + NetworkX + checksum manifest

The demo shell does not access FAISS, SQLite, bm25s, or NetworkX directly. HybridMind owns persistence, retrieval, cache invalidation, checksums, and snapshot rotation.

| User action | API integration | Behaviour |
| --- | --- | --- |
| Save a note, turn, ticket, lesson, or decision | POST /nodes or POST /bulk/nodes | Store text and metadata; HybridMind embeds it at 4096 dimensions and creates sentence chunks. |
| Import a curated corpus | POST /bulk/import or POST /bulk/unstructured | Validate provenance and tag all seed data as synthetic. |
| Link people, accounts, concepts, decisions, or tasks | POST /edges | Use typed edges such as belongs_to, about, decided_by, blocks, supports, and next_turn. |
| Retrieve grounded context | POST /search/hybrid | Tri-signal RRF (dense, BM25, graph) plus configured cross-encoder reranking; use include_images only for multimodal. |
| Explain a relationship | GET /search/path/{source_id}/{target_id}; GET /edges/node/{id} | Render a compact trail and never invent a path. |
| Inspect/remove a memory | GET /nodes/{id}; DELETE /nodes/{id} | Surface a clear, auditable user action. |
| Diagnostics and persistence | GET /health; GET /ready; GET /search/stats; POST /snapshot | Gate demos on readiness and show snapshot identity. |
| Add an image | POST /nodes/image | Requires the remote image-embedding endpoint; its caption still enters the mandatory 4096-dim text index. |

The answer layer receives only the question and retrieved evidence. It returns source node IDs and short excerpts with every answer. When evidence is absent, it explicitly says so. Ranked scores are diagnostic signals, not user-facing confidence probabilities.

### Schema and isolation

Each node has workspace_id plus a domain isolation key: tenant_id, customer_id, learner_id, or project_id. Any real deployment applies this scope as a server-side authorization filter; client filtering is never a security boundary.

    {
      "workspace_id": "support-acme-demo",
      "tenant_id": "acme",
      "source_type": "ticket",
      "timestamp": "2026-08-12T10:30:00Z",
      "sensitivity": "synthetic-internal",
      "provenance": "seed-corpus-v1"
    }

Use sessionId for multi-turn data so current temporal edges are created. Domain meaning belongs in typed edges; node text remains the source record.

## Multimodal design

The field notebook uses the remote ColQwen2.5-compatible image service for patch vectors stored by the visual store. The image also becomes a normal HybridMind node whose caption is embedded through mandatory 4096-dimensional TEI. Text search can therefore locate captioned images; visual MaxSim can improve visual matching when configured; graph edges connect the image to observations and tasks.

If the image service is unavailable, disable image ingestion with an explicit unavailable state. Do not claim the image was indexed and do not relax the 4096 text-vector contract.

## Data, privacy, and safety

- Seed corpora are synthetic, fictional, non-sensitive, and visibly labelled in the UI and provenance metadata.
- Every demo has its own workspace and .mind snapshot. Never reuse a support/customer workspace for assistant or tutor data.
- Ingest the minimum data needed for the journey. Do not use secrets, payment details, health data, student records, or unnecessary identifiers in public demonstrations.
- Make retention, export, reset, and deletion visible. Explain logical deletion, index compaction, and backup retention before any real-user pilot.
- Validate .mind manifests and checksums before restore; never expose an unverified snapshot.
- Keep TEI, image-service, Z.AI, RunPod, and research-proxy credentials server-side. Logs are redacted and avoid retaining sensitive prompts unless a deployment policy permits it.

## Success metrics

### Reliability gates

- 100% of launches verify a reachable TEI service and a 4096-dimensional sample vector.
- Zero silent fallbacks and zero mixed-dimension index events.
- 100% of answerable scripted questions show at least one supporting source node; unanswerable prompts visibly report insufficient evidence.
- 100% of workspace-isolation tests reject cross-workspace retrieval.

### Product evidence

Each corpus has a versioned question/evidence set covering semantic, lexical, temporal, and multi-hop retrieval.

| Measure | Initial target | Method |
| --- | --- | --- |
| Evidence Recall@5 | >= 0.85 | Compare top-five node IDs against annotated evidence. |
| Grounded answer rate | >= 0.90 | Human or Z.AI evaluation verifies the displayed nodes support the response. |
| Multi-hop evidence path rate | >= 0.75 | Required linked facts appear in results or graph path. |
| Median warm retrieval latency | < 2 seconds | Measure request to ranked evidence; report cold-start time separately. |
| Guided completion rate | >= 80% | Observed evaluators complete the journey and locate evidence. |
| Trust signal | >= 70% | Post-demo “would use for context” survey plus free-text reasons. |

Persist benchmark runs in the existing evaluation ledger. Use its confidence intervals and paired permutation tests for configuration comparisons.

## Staged rollout

### Stage 0 — foundation and corpus

Create synthetic/licensed seed corpora, annotations, and isolated .mind workspaces. Validate preflight, readiness, snapshot, reset, and deletion. Build the personal-assistant and project-memory reference corpora first.

**Exit:** scripted journeys work against the live API with verified 4096-dimensional vectors and evidence panels.

### Stage 1 — core interactive demos

Deliver personal assistant, support/CRM, and project-memory shells with fixed prompts, labelled free-query mode, evidence inspection, feedback capture, graph-path view, and reset controls.

**Exit:** reliability gates pass and each corpus has a baseline ledger.

### Stage 2 — professional and tutor variants

Add professional and tutor schemas and curated multi-hop questions while retaining the shared shell. Evaluate whether people can explain why the response was grounded and distinguish evidence from synthesis.

**Exit:** usability tests meet completion and trust targets without vertical-product scope creep.

### Stage 3 — multimodal research preview

Enable the notebook only when the remote image service is configured and validated. Label it experimental; benchmark text-only, visual-only, and combined retrieval separately.

**Exit:** image failure states are clear, cross-modal evidence is inspectable, and the text index still passes the 4096-dimensional invariant.

### Stage 4 — controlled real-data pilot

After separate security and authorization review, invite a small consented group to one isolated domain. Establish retention, deletion, backups, incident response, and feedback handling before the pilot. This is not a general release.

## Acceptance checklist

- [ ] One coherent memory story exists for all six domains.
- [ ] Every answerable scripted question renders attributable evidence.
- [ ] Every text vector and the FAISS index are verified at 4096 dimensions.
- [ ] Service or dimension failures visibly halt the affected demo.
- [ ] Data is synthetic by default and isolated by workspace/domain key.
- [ ] Snapshot integrity, reset, and deletion are demonstrated.
- [ ] Retrieval, latency, groundedness, and trust metrics are captured.
- [ ] The multimodal path is optional, explicit, and separately validated.
