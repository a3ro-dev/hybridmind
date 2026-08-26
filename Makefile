.PHONY: test verify compile check

export PYTHONPATH=$(PWD)

# Canonical offline verification (run inside the activated .venv).
test:
	pytest tests/ -q

# Legacy opt-in feature smoke suite (not part of the default run).
verify:
	pytest verify/ -q

compile:
	python -m compileall -q main.py config.py install.py eval_common.py eval_ledger.py eval_stats.py eval_locomo_retrieval.py eval_longmemeval_retrieval.py eval_musique_retrieval.py api engine storage models cli sdk mcp_server scripts benchmarks tests verify

check: test compile
