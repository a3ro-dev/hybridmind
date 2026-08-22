import assert from "node:assert/strict";
import test from "node:test";

import { HybridMindProvider } from "../../memorybench/memorybench/src/providers/hybridmind/index.ts";

function response(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "Content-Type": "application/json" },
  });
}

function trace(
  overrides: Record<string, unknown> = {},
): Record<string, unknown> {
  return {
    schema_version: "hybridmind.search-execution/v1",
    search_mode: "vector_sparse",
    corpus_generation: 7,
    resolved_config_sha256: "a".repeat(64),
    stages: {
      dense: { requested: true, executed: true, candidates: 1 },
      sparse: { requested: true, executed: true, candidates: 1 },
      graph: { requested: false, executed: false, candidates: 0 },
    },
    ...overrides,
  };
}

test("MemoryBench search fails closed without a corpus scope", async () => {
  const provider = new HybridMindProvider();
  await assert.rejects(
    provider.search("where is it?", { containerTag: "" }),
    /containerTag/,
  );
});

test("MemoryBench search applies its scoped controlled condition", async (t) => {
  const requests: { url: string; body: Record<string, unknown> }[] = [];
  t.mock.method(
    globalThis,
    "fetch",
    async (input: RequestInfo | URL, init?: RequestInit): Promise<Response> => {
      requests.push({
        url: String(input),
        body: JSON.parse(String(init?.body)) as Record<string, unknown>,
      });
      return response({
        results: [{ node_id: "n1" }],
        execution_trace: trace(),
      });
    },
  );

  const provider = new HybridMindProvider();
  const results = (await provider.search("ordinary query", {
    containerTag: "run-17",
    limit: 9,
  })) as Record<string, unknown>[];

  assert.equal(results[0].id, "n1");
  assert.equal(requests.length, 1);
  assert.equal(requests[0].url, "http://127.0.0.1:8000/search/hybrid");
  assert.deepEqual(requests[0].body.filter_metadata, {
    containerTag: "run-17",
  });
  assert.equal(requests[0].body.search_mode, "vector_sparse");
  assert.equal(requests[0].body.route_weights, false);
  assert.equal(requests[0].body.rerank_pool, 0);
});

test("MemoryBench search rejects an unattested execution", async (t) => {
  t.mock.method(globalThis, "fetch", async (): Promise<Response> => {
    return response({
      results: [],
      execution_trace: trace({ search_mode: "hybrid" }),
    });
  });
  const provider = new HybridMindProvider();
  await assert.rejects(
    provider.search("ordinary query", { containerTag: "run-17" }),
    /unexpected search mode/,
  );
});

test("MemoryBench clear deletes only nodes in the requested container", async (t) => {
  const deleted: string[] = [];
  t.mock.method(
    globalThis,
    "fetch",
    async (input: RequestInfo | URL, init?: RequestInit): Promise<Response> => {
      const url = String(input);
      if (!init?.method) {
        return response([
          { id: "keep", metadata: { containerTag: "other" } },
          { id: "delete-a", metadata: { containerTag: "run-17" } },
          { id: "delete-b", metadata: { container_tag: "run-17" } },
        ]);
      }
      deleted.push(url);
      return response({ deleted: true });
    },
  );

  const provider = new HybridMindProvider();
  await provider.clear("run-17");

  assert.deepEqual(deleted, [
    "http://127.0.0.1:8000/nodes/delete-a",
    "http://127.0.0.1:8000/nodes/delete-b",
  ]);
});
