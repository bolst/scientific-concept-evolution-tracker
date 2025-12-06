#!/bin/bash
#SBATCH --job-name=scet_embed
#SBATCH --partition=cpunodes
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --output=logs/embedding_%j.out
#SBATCH --error=logs/embedding_%j.err
#SBATCH --time=24:00:00

# example command: sbatch scripts/generate_embeddings.sh

# ensure we are in the project root
if [ -n "$SLURM_SUBMIT_DIR" ]; then
    cd "$SLURM_SUBMIT_DIR"
else
    cd "$(dirname "$0")/.."
fi

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
