#!/bin/bash -l
#SBATCH -J Detection # Job name
#SBATCH -o /mnt/aiongpfs/users/gbanisetty/cvia/logs/spacecraft_%j.out
#SBATCH -e /mnt/aiongpfs/users/gbanisetty/cvia/logs/spacecraft_%j.err
#SBATCH --time=05:00:00 # Extend runtime for inference
#SBATCH --nodes=1 # Single node
#SBATCH --ntasks=2 # Two tasks (processes) for DDP
#SBATCH --cpus-per-task=10 # Threads for data loading
#SBATCH --mem=128G # System memory
#SBATCH --gres=gpu:2 # ✅ Requests 2 GPUs
#SBATCH -p gpu # ✅ Use 'gpu' partition

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
    # CRITICAL FIX: Update the default path to include the new subfolder
    PYTHON_SCRIPT="detection_detr/main_tensor.py"
fi

echo "=============================="
echo "WORKING DIRECTORY: $(pwd)"
echo "STARTING PYTHON SCRIPT: $PYTHON_SCRIPT"
echo "ARGUMENTS: $SCRIPT_ARGS"
echo "=============================="

# =======================================================
# 4️⃣  Run Training + Inference (DDP using torchrun)
# =======================================================
echo "🚀 Starting detr Pipeline with torchrun (DDP)"
echo "GPU Info:"
nvidia-smi || echo "⚠️ No GPU info (running CPU fallback?)"

# KEY FIX: Use torchrun to launch one process per task ($SLURM_NTASKS is 2)
# torchrun handles setting DDP environment variables for your PyTorch code.
torchrun \
    --nproc_per_node=$SLURM_NTASKS \
    "$PYTHON_SCRIPT" $SCRIPT_ARGS

EXIT_CODE=$?

# =======================================================
# 5️⃣  Wrap-Up
# =======================================================
if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ Job completed successfully!"
else
    echo "❌ Job failed with exit code $EXIT_CODE"
fi

echo "📦 Results stored under: runs_tensor/train_detr34/"
echo "🕒 Finished at: $(date)"