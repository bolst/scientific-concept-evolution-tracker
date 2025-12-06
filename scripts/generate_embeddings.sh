#!/bin/bash

# ensure we are in the project root
cd "$(dirname "$0")/.."

echo "Current directory: $(pwd)"

MUTE=false

# Check for mute argument
for arg in "$@"; do
    if [ "$arg" == "-m" ] || [ "$arg" == "--mute" ]; then
        MUTE=true
        break
    fi
done

# run script
echo "Running generate_embeddings.py..."
uv run scripts/generate_embeddings.py --batch-size 2000 --num-workers 16

EXIT_CODE=$?

# send text when done
if [ $EXIT_CODE -eq 0 ]; then
    echo "Success!"
    if [ "$MUTE" = false ]; then
        echo "Sending notification text..."
        uv run scripts/send_text.py "Scientific Concept Evolution Tracker: Embedding generation job completed successfully."
    fi
else
    echo "Failure!"
    if [ "$MUTE" = false ]; then
        echo "Sending notification text..."
        uv run scripts/send_text.py "Scientific Concept Evolution Tracker: Embedding generation job FAILED with exit code $EXIT_CODE."
    fi
fi

exit $EXIT_CODE
