import type {
  Provider,
  ProviderConfig,
  IngestOptions,
  IngestResult,
  SearchOptions,
  IndexingProgressCallback,
} from "../../types/provider"
import type { UnifiedSession } from "../../types/unified"
import { logger } from "../../utils/logger"
import crypto from "crypto"

// ─── Query Router (mirrors engine/query_router.py) ───────────────────────────
const TEMPORAL_RE =
  /\b(when|how long|before|after|during|date|year|month|first time|last time|how many (?:days|months|years)|\d{4})\b/i
const MULTIHOP_RE = /\b(relationship between|how.*connected|both|common between|link between)\b/i
const ENTITY_RE = /\b(who is|what is|where is|who was|what was|tell me about)\b/i

interface RouteResult {
  type: "temporal" | "multihop" | "entity" | "default"
  metadataFilter: Record<string, unknown> | null
}

function routeQuery(question: string): RouteResult {
  if (TEMPORAL_RE.test(question)) return { type: "temporal", metadataFilter: null }
  if (MULTIHOP_RE.test(question)) return { type: "multihop", metadataFilter: null }
  if (ENTITY_RE.test(question))
    return { type: "entity", metadataFilter: { type: "extracted_fact" } }
  return { type: "default", metadataFilter: null }
}

// ─── Provider ────────────────────────────────────────────────────────────────

export class HybridMindProvider implements Provider {
  name = "hybridmind"
  concurrency = { default: 1, ingest: 3, indexing: 50, answer: 1, evaluate: 1 }
  private baseUrl: string = process.env.HYBRIDMIND_BASE_URL || "http://127.0.0.1:8000"
  private apiKey: string = process.env.HYBRIDMIND_API_KEY || ""

  private headers(): Record<string, string> {
    return {
      "Content-Type": "application/json",
      ...(this.apiKey ? { "X-HybridMind-API-Key": this.apiKey } : {}),
    }
  }

  private requireContainerTag(containerTag: string): string {
    const tag = containerTag.trim()
    if (!tag) {
      throw new Error("HybridMind requires a non-empty containerTag for corpus isolation")
    }
    return tag
  }

  private validateExecutionTrace(data: Record<string, unknown>): void {
    const trace = data.execution_trace as Record<string, unknown> | undefined
    if (!trace || trace.schema_version !== "hybridmind.search-execution/v1") {
      throw new Error("HybridMind search response is missing the v1 execution trace")
    }
    if (trace.search_mode !== "vector_sparse") {
      throw new Error(`HybridMind executed unexpected search mode: ${String(trace.search_mode)}`)
    }
    if (!Number.isInteger(trace.corpus_generation) || (trace.corpus_generation as number) < 0) {
      throw new Error("HybridMind execution trace has an invalid corpus generation")
    }
    if (!/^[a-f0-9]{64}$/.test(String(trace.resolved_config_sha256 ?? ""))) {
      throw new Error("HybridMind execution trace has an invalid resolved config hash")
    }

    const stages = trace.stages as Record<string, Record<string, unknown>> | undefined
    for (const stageName of ["dense", "sparse"] as const) {
      const stage = stages?.[stageName]
      if (!stage || stage.requested !== true || stage.executed !== true) {
        throw new Error(`HybridMind did not execute requested ${stageName} retrieval`)
      }
    }
    const graph = stages?.graph
    if (!graph || graph.requested !== false || graph.executed !== false) {
      throw new Error("HybridMind vector_sparse condition unexpectedly executed graph retrieval")
    }
  }

  prompts = {
    answerPrompt: (question: string, context: unknown[], questionDate?: string): string => {
      // Format context snippets with clear numbering and date info
      const contextSnippets = (context as Record<string, unknown>[])
        .slice(0, 15)
        .map((item, i) => {
          const text = (item.text as string) || JSON.stringify(item)
          const meta = (item.metadata as Record<string, unknown>) || {}
          const date = meta.timestamp || meta.date || meta.createdAt || ""
          const type = meta.type || ""
          let snippet = `[${i + 1}] ${text}`
          if (date) snippet += `\n    Date: ${date}`
          if (type === "extracted_fact") snippet += `\n    (extracted fact)`
          return snippet
        })
        .join("\n\n")

      const route = routeQuery(question)

      // Detect inference questions ("Would X...?", "Could X...?", "Is X likely to...?")
      const isInferenceQ = /\b(would|could|is .* likely|might|can)\b/i.test(question)

      let typeInstructions = ""
      if (isInferenceQ) {
        typeInstructions = `
TYPE-SPECIFIC (inference / hypothetical):
- The question asks you to reason from facts. INFER and commit to "Yes" or "No" plus a short reason from the context.
- Do NOT say "I don't know" if the context contains relevant facts — use them to reason.`
      } else if (route.type === "temporal") {
        typeInstructions = `
TYPE-SPECIFIC (temporal — date/time):
- Give the exact date, period, or relative time (e.g. "next month", "the week before June 9") as it appears in the context.
- For "how long" questions: subtract the two dates shown in context to compute the duration.
- NEVER invent or compute a specific date that is not shown in the context.`
      } else if (route.type === "multihop") {
        typeInstructions = `
TYPE-SPECIFIC (multi-hop):
- Chain facts across snippets step-by-step, then state only the final answer.`
      } else if (route.type === "entity") {
        typeInstructions = `
TYPE-SPECIFIC (entity):
- Prefer answers from snippets marked (extracted fact).`
      }

      return `You are answering questions about a long-term conversation between two people.

RETRIEVED CONTEXT:
${contextSnippets}

Question: ${question}
${questionDate ? `Reference Date: ${questionDate}` : ""}

INSTRUCTIONS:
1. Give a SHORT, DIRECT answer (a few words or one sentence).
2. Factual (who/what/where): scan ALL snippets and use the one that contains the answer — even if it's near the end of the list.
3. Date/time (when): give the date, period, or relative time EXACTLY as it appears in the context (e.g. "the week before June 9", "next month", "7 May 2023"). Do NOT convert or recompute dates.
4. Duration (how long): subtract the two dates shown in the context.
5. Multi-part: combine facts from multiple snippets.
6. Inference ("Would X…?", "Is X likely to…?", "Could X…?"): reason from the retrieved facts and commit to a clear answer. Do NOT say "I don't know" when the context contains relevant information.
7. If ANY snippet contains information that could answer the question — even partially — USE IT. Do NOT say "I don't know" just because the answer isn't prominent.
8. Say "I don't know" ONLY if ALL context snippets are completely irrelevant to the question topic.
9. No extra explanation — just the answer.
10. [DATE: …] prefixes are reliable date markers; use them.
${typeInstructions}

Answer:`
    },
  }

  async initialize(config: ProviderConfig): Promise<void> {
    if (config.baseUrl) this.baseUrl = config.baseUrl as string
    if (config.apiKey) this.apiKey = config.apiKey
    // The detailed health endpoint performs a real remote embedding and can
    // stall during a serverless cold start. Readiness verifies the live API
    // without turning provider initialization into an embedding request.
    const res = await fetch(`${this.baseUrl}/ready`, { headers: this.headers() })
    if (!res.ok) throw new Error(`HybridMind not healthy at ${this.baseUrl}: ${res.status}`)
    logger.info(`Initialized HybridMind provider at ${this.baseUrl}`)
  }

  async ingest(sessions: UnifiedSession[], options: IngestOptions): Promise<IngestResult> {
    const documentIds: string[] = []
    const containerTag = this.requireContainerTag(options.containerTag)

    try {
      for (const session of sessions) {
        const sessionDate = session.metadata?.date as string | undefined
        const bulkNodes: Record<string, unknown>[] = []

        for (const msg of session.messages) {
          const role = msg.role === "user" ? "human" : "ai"
          const speaker =
            ((msg as unknown as Record<string, unknown>).speaker as string | undefined) || role
          const dateStr =
            ((msg as unknown as Record<string, unknown>).timestamp as string) ?? sessionDate ?? ""
          const text = dateStr
            ? `[DATE: ${dateStr}] [SPEAKER: ${speaker}] ${msg.content}`
            : `[SPEAKER: ${speaker}] ${msg.content}`

          const metadata: Record<string, unknown> = {
            ...session.metadata,
            session_id: session.sessionId,
            sessionId: session.sessionId,
            containerTag,
            role,
            timestamp: (msg as unknown as Record<string, unknown>).timestamp ?? sessionDate ?? "",
          }

          // Generate client-side UUID for bulk indexing
          const customId = crypto.randomUUID()
          documentIds.push(customId)

          bulkNodes.push({
            id: customId,
            text,
            metadata,
          })
        }

        // 1. Bulk ingest raw conversation turns. The server compensates the
        // whole batch if any authoritative or derived-index mutation fails.
        if (bulkNodes.length > 0) {
          const res = await fetch(`${this.baseUrl}/bulk/nodes`, {
            method: "POST",
            headers: this.headers(),
            body: JSON.stringify({
              nodes: bulkNodes,
              generate_embeddings: true,
            }),
          })
          if (!res.ok) {
            const body = await res.text()
            throw new Error(`HybridMind bulk ingest failed: ${res.status} ${body}`)
          }
          const result = (await res.json()) as {
            success?: boolean
            created?: number
            failed?: number
          }
          if (
            result.success !== true ||
            result.failed !== 0 ||
            result.created !== bulkNodes.length
          ) {
            throw new Error("HybridMind bulk ingest returned a partial or malformed receipt")
          }
        }

        // 2. Extract and store facts for this session (LLM call at ingest time
        // only). A failed extraction invalidates the benchmark corpus.
        const turns = session.messages.map((msg) => ({
          speaker:
            ((msg as unknown as Record<string, unknown>).speaker as string | undefined) ||
            (msg.role === "user" ? "human" : "ai"),
          text: msg.content,
          date:
            ((msg as unknown as Record<string, unknown>).timestamp as string) ?? sessionDate ?? "",
        }))

        const factsRes = await fetch(`${this.baseUrl}/ingest/session-facts`, {
          method: "POST",
          headers: this.headers(),
          body: JSON.stringify({
            session_id: session.sessionId,
            turns,
            container_tag: containerTag,
          }),
        })

        if (!factsRes.ok) {
          const body = await factsRes.text()
          throw new Error(
            `HybridMind session-facts failed for ${session.sessionId}: ${factsRes.status} ${body}`
          )
        }
        const factsData = (await factsRes.json()) as {
          facts_extracted?: number
          node_ids?: string[]
        }
        if (
          !Number.isInteger(factsData.facts_extracted) ||
          !Array.isArray(factsData.node_ids) ||
          factsData.node_ids.some((id) => typeof id !== "string")
        ) {
          throw new Error("HybridMind session-facts returned a malformed receipt")
        }
        documentIds.push(...factsData.node_ids)
        logger.info(`Session ${session.sessionId}: ${factsData.facts_extracted} facts extracted`)
      }
    } catch (error) {
      try {
        await this.clear(containerTag)
      } catch (cleanupError) {
        throw new AggregateError(
          [error, cleanupError],
          "HybridMind ingestion failed and scoped compensation also failed"
        )
      }
      throw error
    }

    return { documentIds }
  }

  async awaitIndexing(
    result: IngestResult,
    _containerTag: string,
    onProgress?: IndexingProgressCallback
  ): Promise<void> {
    onProgress?.({
      completedIds: result.documentIds,
      failedIds: [],
      total: result.documentIds.length,
    })
  }

  async search(query: string, options: SearchOptions): Promise<unknown[]> {
    const topK = options.limit || 15
    const containerTag = this.requireContainerTag(options.containerTag)
    const route = routeQuery(query)
    const scopeFilter: Record<string, unknown> = { containerTag }

    const body: Record<string, unknown> = {
      query_text: query,
      top_k: topK,
      min_score: 0.0,
      search_mode: "vector_sparse",
      vector_weight: 0.6,
      bm25_boost_weight: 0.4,
      graph_weight: 0.0,
      route_weights: false,
      rerank_pool: 0,
      track_access: false,
      filter_metadata: scopeFilter,
    }

    // Entity questions prefer extracted facts, but fallback remains scoped to
    // the same corpus if that representation has no candidates.
    if (route.type === "entity" && route.metadataFilter) {
      body.filter_metadata = { ...scopeFilter, ...route.metadataFilter }
    }

    const res = await fetch(`${this.baseUrl}/search/hybrid`, {
      method: "POST",
      headers: this.headers(),
      body: JSON.stringify(body),
    })

    if (!res.ok) {
      const errBody = await res.text()
      throw new Error(`HybridMind search failed: ${res.status} ${errBody}`)
    }

    const data = (await res.json()) as Record<string, unknown> & { results?: unknown[] }
    this.validateExecutionTrace(data)
    let results = data.results || []

    if (route.type === "entity" && results.length === 0) {
      const fallbackRes = await fetch(`${this.baseUrl}/search/hybrid`, {
        method: "POST",
        headers: this.headers(),
        body: JSON.stringify({ ...body, filter_metadata: scopeFilter }),
      })
      if (!fallbackRes.ok) {
        const errBody = await fallbackRes.text()
        throw new Error(`HybridMind entity fallback failed: ${fallbackRes.status} ${errBody}`)
      }
      const fallbackData = (await fallbackRes.json()) as Record<string, unknown> & {
        results?: unknown[]
      }
      this.validateExecutionTrace(fallbackData)
      results = fallbackData.results || []
    }

    // Normalize: MemoryBench expects result.id — HybridMind returns result.node_id
    const normalizedResults = results.map((r: unknown) => {
      const result = r as Record<string, unknown>
      return {
        ...result,
        id: result.id ?? result.node_id ?? null,
      }
    })

    return normalizedResults
  }

  async clear(containerTag: string): Promise<void> {
    const tag = this.requireContainerTag(containerTag)
    const pageSize = 1000
    const matchingIds: string[] = []

    for (let skip = 0; ; skip += pageSize) {
      const res = await fetch(`${this.baseUrl}/nodes?skip=${skip}&limit=${pageSize}`, {
        headers: this.headers(),
      })
      if (!res.ok) {
        const body = await res.text()
        throw new Error(`HybridMind list for clear failed: ${res.status} ${body}`)
      }
      const nodes = (await res.json()) as {
        id: string
        metadata?: Record<string, unknown>
      }[]
      for (const node of nodes) {
        const nodeTag = node.metadata?.containerTag ?? node.metadata?.container_tag
        if (nodeTag === tag) matchingIds.push(node.id)
      }
      if (nodes.length < pageSize) break
    }

    for (const nodeId of matchingIds) {
      const res = await fetch(`${this.baseUrl}/nodes/${encodeURIComponent(nodeId)}`, {
        method: "DELETE",
        headers: this.headers(),
      })
      if (!res.ok && res.status !== 404) {
        const body = await res.text()
        throw new Error(`HybridMind clear failed for ${nodeId}: ${res.status} ${body}`)
      }
    }
    logger.info(`HybridMind: cleared ${matchingIds.length} nodes for container ${tag}`)
  }
}
