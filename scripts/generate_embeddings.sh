#!/bin/bash

# ensure we are in the project root
cd "$(dirname "$0")/.."

echo "Current directory: $(pwd)"

# run script
echo "Running generate_embeddings.py..."
uv run scripts/generate_embeddings.py --batch-size 100

EXIT_CODE=$?

# send text when done
if [ $EXIT_CODE -eq 0 ]; then
    echo "Success!"
    uv run scripts/send_text.py "Scientific Concept Evolution Tracker: Embedding generation job completed successfully."
else
    echo "Failure!"
    uv run scripts/send_text.py "Scientific Concept Evolution Tracker: Embedding generation job FAILED with exit code $EXIT_CODE."
fi

exit $EXIT_CODE
