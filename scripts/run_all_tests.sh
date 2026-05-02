#!/bin/bash
set -e

echo "Running Unit Tests..."
python -m pytest tests/

echo "All tests passed successfully."
