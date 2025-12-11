#!/bin/bash
#SBATCH --job-name=scet_embed
#SBATCH --array=0-9                 # Run 10 parallel jobs
#SBATCH --output=logs/embed_%a.log  # Log file per array task
#SBATCH --cpus-per-task=8           # CPUs per task
#SBATCH --mem=32G                   # Memory per task
#SBATCH --time=72:00:00             # Max runtime

mkdir -p logs

echo "Starting task $SLURM_ARRAY_TASK_ID..."

# Run the embedding script with sharding arguments
# We use SLURM_ARRAY_TASK_COUNT if available, otherwise calculate from array size
if [ -z "$SLURM_ARRAY_TASK_COUNT" ]; then
    # Fallback if count variable is not set (older Slurm)
    # Assumes array is 0-N
    NUM_SHARDS=$(scontrol show job $SLURM_JOB_ID | grep "ArrayTaskId" | awk -F'-' '{print $2}' | awk '{print $1}')
    if [ -z "$NUM_SHARDS" ]; then
        NUM_SHARDS=10 # Default fallback
    else
        NUM_SHARDS=$((NUM_SHARDS + 1))
    fi
else
    NUM_SHARDS=$SLURM_ARRAY_TASK_COUNT
fi

echo "Running shard $SLURM_ARRAY_TASK_ID of $NUM_SHARDS"

uv run src/embed_milvus.py \
    --shard-id $SLURM_ARRAY_TASK_ID \
    --num-shards $NUM_SHARDS \
    --batch-size 1000

EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
    echo "Task $SLURM_ARRAY_TASK_ID completed successfully."
    # Only send text for the first task to avoid spam
    if [ "$SLURM_ARRAY_TASK_ID" -eq 0 ]; then
        uv run src/send_text.py "SCET: Embedding generation job (Task 0) completed successfully."
    fi
else
    echo "Task $SLURM_ARRAY_TASK_ID failed with exit code $EXIT_CODE."
    # Only send text for the first task to avoid spam
    if [ "$SLURM_ARRAY_TASK_ID" -eq 0 ]; then
        uv run src/send_text.py "SCET: Embedding generation job (Task 0) FAILED with exit code $EXIT_CODE."
    fi
    exit $EXIT_CODE
fi
