.PHONY: test test-slow benchmark report full-eval

export PYTHONPATH=$(PWD)

test:
	pytest tests/ -m "not slow" -v

test-slow:
	pytest tests/ -v

benchmark:
	python3 benchmarks/ablation_runner.py

report:
	python3 benchmarks/generate_report.py

full-eval: benchmark report test
