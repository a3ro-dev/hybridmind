"""Quarantined legacy sample scorer.

This historical helper called Z.AI directly, embedded host-specific checkpoint
paths, omitted provider/spend gates, and treated provider or JSON failures as
incorrect answers.  Those properties make its output ineligible as benchmark
evidence and unsafe to run accidentally.

Use the canonical evaluator instead, for example::

    python eval_locomo_retrieval.py --execute --with-answers \
        --max-queries 20 --max-llm-calls 40 \
        --max-input-tokens ... --max-output-tokens ... \
        --max-estimated-spend-usd ... \
        --input-cost-per-million-tokens ... \
        --output-cost-per-million-tokens ...

That path uses ``engine.llm_client``, enforces explicit resource ceilings,
records provider failures separately from wrong answers and abstentions, and
writes an immutable provenance ledger.
"""


def main() -> int:
    raise SystemExit(
        "scripts/score_sample_20.py is quarantined: it cannot produce valid "
        "benchmark evidence. Use eval_locomo_retrieval.py with --execute, "
        "--with-answers, and explicit budget/pricing ceilings."
    )


if __name__ == "__main__":
    main()
