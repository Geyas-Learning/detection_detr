#!/bin/bash -l
#SBATCH -J Segmentation # Job name
#SBATCH -o /mnt/aiongpfs/users/gbanisetty/cvia/logs/spacecraft_%j.out
#SBATCH -e /mnt/aiongpfs/users/gbanisetty/cvia/logs/spacecraft_%j.err
#SBATCH --time=30:00:00 # Extend runtime for inference
#SBATCH --nodes=1 # Single node
#SBATCH --ntasks=1 # ✅ Correct: Request 1 task for single-GPU
#SBATCH --cpus-per-task=24 # Threads for data loading
#SBATCH --mem=128G # System memory
#SBATCH --gres=gpu:1 # ✅ Correct: Request 1 GPU
#SBATCH -p gpu # Use 'gpu' partition

# =======================================================
# 1️⃣  Environment Setup
# =======================================================
module --force purge
source ~/miniconda3/etc/profile.d/conda.sh

ENV_PATH="/mnt/aiongpfs/users/gbanisetty/cvia/detr_env"
echo "🔹 Activating conda environment at: $ENV_PATH"
conda activate "$ENV_PATH"

# =======================================================
# 2️⃣  Project Setup
# =======================================================
PROJECT_DIR="/mnt/aiongpfs/users/gbanisetty/cvia/project_detr"
cd "$PROJECT_DIR" || { echo "❌ ERROR: Cannot cd to $PROJECT_DIR"; exit 1; }

RUN_ID=$(date +"%Y%m%d_%H%M%S")
RUN_NAME="run_${RUN_ID}"
echo "📅 Run ID: $RUN_ID"

# =======================================================
# 3️⃣  Python Script + Args
# =======================================================
PYTHON_SCRIPT="$1"
shift
SCRIPT_ARGS="$@"

if [ -z "$PYTHON_SCRIPT" ]; then
    PYTHON_SCRIPT="detection_detr/main_segm.py"
fi

echo "=============================="
echo "WORKING DIRECTORY: $(pwd)"
echo "STARTING PYTHON SCRIPT: $PYTHON_SCRIPT"
echo "ARGUMENTS: $SCRIPT_ARGS"
echo "=============================="

# =======================================================
# 4️⃣  Run Training + Inference (Single GPU)
# =======================================================
echo "🚀 Starting DETR Segmentation Pipeline (Single GPU)" # ✅ Updated text
echo "GPU Info:"
nvidia-smi || echo "⚠️ No GPU info (running CPU fallback?)"

# ✅ CRITICAL FIX: Direct Python execution for single GPU
python "$PYTHON_SCRIPT" $SCRIPT_ARGS

EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ Job completed successfully!"
else
    echo "❌ Job failed with exit code $EXIT_CODE"
fi

echo "🕒 Finished at: $(date)"