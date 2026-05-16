#!/bin/bash
# =============================================================================
#  AdvSplIoT — Air-Quality Prediction with LLM Explanation
#  Sapelo2 B200 submission script.
#
#  Pipeline (one job, one node, one GPU):
#      1. Load + preprocess Beijing PRSA PM2.5 dataset
#      2. mRMR + Random-Forest feature selection
#      3. Train baseline LSTM
#      4. ISSA hyper-parameter search (Improved Sparrow Search)
#      5. Re-train LSTM at the best hyper-parameters
#      6. Naive baselines (persistence, AR(24))
#      7. Qwen2.5-7B-Instruct → grounded JSON explanations for N test rows
#      8. Hallucination detection (H1..H6)
#      9. Interactive QA examples
#     10. Auto-generated final_report.md
#
#  Usage:
#      sbatch submit.sh
#
#  Outputs (everything saved by run_pipeline.py):
#      ${RESULTS_DIR}/dataset_stats.json
#      ${RESULTS_DIR}/features/{mrmr_log,mrmr_selected,rf_importance}.csv
#      ${RESULTS_DIR}/models/{metrics_summary,test_predictions,issa_log,...}.csv
#      ${RESULTS_DIR}/llm/{explanations,hallucination_flags,qa_examples}.csv
#      ${RESULTS_DIR}/plots/*.png
#      ${RESULTS_DIR}/final_report.md
# =============================================================================
#SBATCH --job-name=advspliot_air
#SBATCH --partition=iai_B200_p
#SBATCH --qos=iai_b200_p_qos
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=192G
#SBATCH --time=08:00:00
#SBATCH --output=/scratch/sg41479/AdvSplIoT/slurm/advspliot_%j.out
#SBATCH --error=/scratch/sg41479/AdvSplIoT/slurm/advspliot_%j.err
#SBATCH --mail-user=sg41479@uga.edu
#SBATCH --mail-type=END,FAIL

set -euo pipefail

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
cd "${SLURM_SUBMIT_DIR:-$(dirname "$(readlink -f "$0")")}"
REPO_DIR="$(pwd)"

mkdir -p "${REPO_DIR}/slurm"

RUN_ID="${RUN_ID:-${SLURM_JOB_ID:-$(date +%Y%m%d_%H%M%S)}}"
RESULTS_DIR="${RESULTS_DIR:-${REPO_DIR}/results/${RUN_ID}}"
mkdir -p "${RESULTS_DIR}"

# ---------------------------------------------------------------------------
# Python environment — re-use the prebuilt gpt4_eval venv (torch 2.11+cu130,
# transformers 5.4, accelerate 1.13, sklearn, matplotlib, scipy, safetensors).
# Override VENV_DIR to point at a different env if needed.
# ---------------------------------------------------------------------------
VENV_DIR="${VENV_DIR:-/scratch/sg41479/venvs/gpt4_eval}"
PYTHON_BIN="${VENV_DIR}/bin/python"

# Hugging Face cache containing Qwen2.5-7B-Instruct weights.
HF_HOME="${HF_HOME:-/scratch/sg41479/hf-cache}"
TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}}"
HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
LLM_MODEL_PATH="${LLM_MODEL_PATH:-${HF_HOME}/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28}"

# ---------------------------------------------------------------------------
# Pipeline knobs (override via env before sbatch if needed)
# ---------------------------------------------------------------------------
EPOCHS_BASELINE="${EPOCHS_BASELINE:-10}"
EPOCHS_ISSA="${EPOCHS_ISSA:-10}"
ISSA_ITERS="${ISSA_ITERS:-6}"
ISSA_POP="${ISSA_POP:-8}"
ISSA_FITNESS_EPOCHS="${ISSA_FITNESS_EPOCHS:-1}"
LOOKBACK="${LOOKBACK:-60}"
TRAIN_FRAC="${TRAIN_FRAC:-0.72}"
VAL_FRAC="${VAL_FRAC:-0.08}"
SEED="${SEED:-42}"

N_EXPLANATIONS="${N_EXPLANATIONS:-100}"
N_QA_SAMPLES="${N_QA_SAMPLES:-5}"

LLM_DTYPE="${LLM_DTYPE:-bfloat16}"
LLM_MAX_NEW_TOKENS="${LLM_MAX_NEW_TOKENS:-320}"
LLM_TEMPERATURE="${LLM_TEMPERATURE:-0.3}"
NUM_TOL_PCT="${NUM_TOL_PCT:-1.0}"
LOGPROB_THRESHOLD="${LOGPROB_THRESHOLD:--2.0}"

CSV_PATH="${CSV_PATH:-${REPO_DIR}/PRSA_data.csv}"

# ---------------------------------------------------------------------------
# Re-submit if not inside an allocation
# ---------------------------------------------------------------------------
if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  echo "================================================================="
  echo "  Submitting AdvSplIoT air-quality job"
  echo "================================================================="
  echo "  REPO_DIR    = ${REPO_DIR}"
  echo "  RESULTS_DIR = ${RESULTS_DIR}"
  echo "  VENV_DIR    = ${VENV_DIR}"
  echo "  HF_HOME     = ${HF_HOME}"
  echo "  LLM_MODEL_PATH = ${LLM_MODEL_PATH}"
  echo "  N_EXPLANATIONS = ${N_EXPLANATIONS}"
  echo "================================================================="
  EXPORT_VARS=(
    REPO_DIR RESULTS_DIR VENV_DIR HF_HOME TRANSFORMERS_CACHE HF_HUB_OFFLINE
    LLM_MODEL_PATH EPOCHS_BASELINE EPOCHS_ISSA ISSA_ITERS ISSA_POP
    ISSA_FITNESS_EPOCHS LOOKBACK TRAIN_FRAC VAL_FRAC SEED
    N_EXPLANATIONS N_QA_SAMPLES LLM_DTYPE LLM_MAX_NEW_TOKENS LLM_TEMPERATURE
    NUM_TOL_PCT LOGPROB_THRESHOLD CSV_PATH
  )
  EXPORT_ARG="ALL"
  for v in "${EXPORT_VARS[@]}"; do
    if [[ -n "${!v:-}" ]]; then
      EXPORT_ARG="${EXPORT_ARG},${v}=${!v}"
    fi
  done
  JOB_ID=$(sbatch --parsable --export="${EXPORT_ARG}" "$0")
  echo "Submitted job: ${JOB_ID}"
  echo "Watch:   squeue -u \$USER -j ${JOB_ID}"
  echo "Logs:    ${REPO_DIR}/slurm/advspliot_${JOB_ID}.out"
  echo "Results: ${RESULTS_DIR}"
  exit 0
fi

# ---------------------------------------------------------------------------
# Inside the SLURM allocation
# ---------------------------------------------------------------------------
ml purge
# Load CUDA 12.x for the B200 (we already use a torch wheel built against
# CUDA 13.0 via cu130; the driver on the B200 node is ≥ 12.8 which is
# binary-compatible with the cu13 wheel).
ml CUDA/12.8.0 2>/dev/null || ml CUDA/12.4.0 2>/dev/null || ml CUDA/11.8.0 || true

# Sanity: env exists
if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "[boot] ERROR: python not found at ${PYTHON_BIN}" >&2
  exit 2
fi

export PATH="${VENV_DIR}/bin:${PATH}"
export LD_LIBRARY_PATH="${VENV_DIR}/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="${REPO_DIR}:${PYTHONPATH:-}"

# Hugging Face: stay offline so the loader uses the cached Qwen weights only.
export HF_HOME TRANSFORMERS_CACHE HF_HUB_OFFLINE

# CPU thread pinning — keep BLAS sane.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-4}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-4}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

# Sanity-print: GPU + python + library versions.
"${PYTHON_BIN}" - <<'PY'
import torch, transformers, sys
print(f"[boot] python={sys.version.split()[0]}")
print(f"[boot] torch={torch.__version__} cuda={torch.version.cuda}")
print(f"[boot] transformers={transformers.__version__}")
print(f"[boot] cuda_available={torch.cuda.is_available()}")
for i in range(torch.cuda.device_count()):
    cap = torch.cuda.get_device_capability(i)
    mem = torch.cuda.get_device_properties(i).total_memory / 1e9
    print(f"[boot]   GPU{i}: {torch.cuda.get_device_name(i)} sm_{cap[0]}{cap[1]} {mem:.1f} GB")
PY

echo "================================================================="
echo "  AdvSplIoT — air-quality + LLM explanation pipeline"
echo "================================================================="
echo "  JOB_ID      = ${SLURM_JOB_ID:-NONE}"
echo "  REPO_DIR    = ${REPO_DIR}"
echo "  RESULTS_DIR = ${RESULTS_DIR}"
echo "  CSV_PATH    = ${CSV_PATH}"
echo "  EPOCHS      = baseline=${EPOCHS_BASELINE} issa=${EPOCHS_ISSA}"
echo "  ISSA        = pop=${ISSA_POP} iters=${ISSA_ITERS} fit_ep=${ISSA_FITNESS_EPOCHS}"
echo "  LLM         = ${LLM_MODEL_PATH}  dtype=${LLM_DTYPE}"
echo "  N_EXPLAN    = ${N_EXPLANATIONS}  N_QA=${N_QA_SAMPLES}"
echo "================================================================="

cd "${REPO_DIR}"

set -x
"${PYTHON_BIN}" -u scripts/run_pipeline.py \
    --csv               "${CSV_PATH}" \
    --results_dir       "${RESULTS_DIR}" \
    --lookback          "${LOOKBACK}" \
    --train_frac        "${TRAIN_FRAC}" \
    --val_frac          "${VAL_FRAC}" \
    --epochs_baseline   "${EPOCHS_BASELINE}" \
    --epochs_issa       "${EPOCHS_ISSA}" \
    --issa_iters        "${ISSA_ITERS}" \
    --issa_pop          "${ISSA_POP}" \
    --issa_fitness_epochs "${ISSA_FITNESS_EPOCHS}" \
    --n_explanations    "${N_EXPLANATIONS}" \
    --n_qa_samples      "${N_QA_SAMPLES}" \
    --device            cuda \
    --seed              "${SEED}" \
    --llm_model_path    "${LLM_MODEL_PATH}" \
    --llm_cache_dir     "${HF_HOME}" \
    --llm_dtype         "${LLM_DTYPE}" \
    --llm_max_new_tokens "${LLM_MAX_NEW_TOKENS}" \
    --llm_temperature   "${LLM_TEMPERATURE}" \
    --num_tol_pct       "${NUM_TOL_PCT}" \
    --logprob_threshold "${LOGPROB_THRESHOLD}"
set +x

echo ""
echo "================================================================="
echo "  Pipeline complete."
echo "  Results: ${RESULTS_DIR}"
echo "  Report:  ${RESULTS_DIR}/final_report.md"
echo "================================================================="
