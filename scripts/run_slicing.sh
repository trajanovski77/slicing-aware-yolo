#!/usr/bin/env bash
# Orchestrate the slicing-aware inference study across two GPUs.
#
# Stages:  prepare_gt -> predict -> eval -> diagnostics -> benchmark -> report
# Targets: <stage>            (all datasets)      e.g.  ./run_slicing.sh predict
#          <stage>_<dataset>  (one dataset)       e.g.  ./run_slicing.sh benchmark_dota
#          full               (gt+predict+eval+diagnostics+report; no benchmark)
#          all                (full + benchmark)
#
# Parallelism model (see plan): predict/eval/diagnostics tolerate GPU sharing, so the two
# datasets run concurrently on GPU 0 and GPU 1. benchmark needs a QUIET GPU for valid
# latency -> one dataset per card, in parallel across cards, serial within a card.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

export YOLO_CONFIG_DIR="${YOLO_CONFIG_DIR:-${ROOT}/.ultralytics}"
if [[ -z "${PYTHON:-}" && -x "${ROOT}/.venv/bin/python" ]]; then PYTHON="${ROOT}/.venv/bin/python"; fi
PYTHON="${PYTHON:-python3}"

# Make CUDA device indices match nvidia-smi's PCI-bus ordering so --device N is deterministic.
export CUDA_DEVICE_ORDER="${CUDA_DEVICE_ORDER:-PCI_BUS_ID}"

DATASETS=(dota ships)
declare -A CFG=( [dota]=configs/slicing/dota.yaml [ships]=configs/slicing/ships.yaml )
# dataset -> card for parallel stages (edit for your machine). We pass --device directly
# instead of pinning via CUDA_VISIBLE_DEVICES, because Ultralytics select_device() overwrites
# CUDA_VISIBLE_DEVICES from the --device value and would otherwise collapse both datasets
# onto one card.
declare -A GPU=( [dota]=1 [ships]=0 )

IMAGE_COUNT="${IMAGE_COUNT:-25}"
WARMUP="${WARMUP:-10}"
ITERATIONS="${ITERATIONS:-100}"
LIMIT="${LIMIT:-0}"                       # >0 for smoke runs

gt()      { "${PYTHON}" scripts/prepare_coco_gt.py --config "${CFG[$1]}"; }
predict() { "${PYTHON}" scripts/predict_slicing.py \
              --config "${CFG[$1]}" --device "${2:-${GPU[$1]}}" $([[ "${LIMIT}" -gt 0 ]] && echo "--limit ${LIMIT}"); }
evaluate(){ "${PYTHON}" scripts/evaluate_slicing.py --config "${CFG[$1]}"; }
diag()    { "${PYTHON}" scripts/slicing_diagnostics.py --config "${CFG[$1]}"; }
bench()   { "${PYTHON}" scripts/benchmark_slicing.py \
              --config "${CFG[$1]}" --device "${2:-${GPU[$1]}}" --image-count "${IMAGE_COUNT}" \
              --warmup "${WARMUP}" --iterations "${ITERATIONS}"; }
report()  { "${PYTHON}" scripts/emit_slicing_tables.py --datasets "${DATASETS[*]}"
            "${PYTHON}" scripts/generate_slicing_figures.py --datasets "${DATASETS[*]}"; }

# Run a per-dataset function across all datasets, one per GPU, in parallel.
parallel_over_datasets() {
  local fn="$1"; local pids=()
  for ds in "${DATASETS[@]}"; do "${fn}" "${ds}" & pids+=("$!"); done
  local rc=0; for pid in "${pids[@]}"; do wait "${pid}" || rc=1; done
  return "${rc}"
}

TARGET="${1:-help}"
case "${TARGET}" in
  prepare_gt)  for ds in "${DATASETS[@]}"; do gt "${ds}"; done ;;
  predict)     parallel_over_datasets predict ;;
  eval)        for ds in "${DATASETS[@]}"; do evaluate "${ds}"; done ;;
  diagnostics) for ds in "${DATASETS[@]}"; do diag "${ds}"; done ;;
  benchmark)   parallel_over_datasets bench ;;   # one dataset per quiet card
  report)      report ;;
  full)        for ds in "${DATASETS[@]}"; do gt "${ds}"; done
               parallel_over_datasets predict
               for ds in "${DATASETS[@]}"; do evaluate "${ds}"; diag "${ds}"; done
               report ;;
  all)         "${BASH_SOURCE[0]}" full
               parallel_over_datasets bench
               report ;;
  predict_*|eval_*|diagnostics_*|benchmark_*)
               ds="${TARGET#*_}"
               [[ -n "${CFG[$ds]:-}" ]] || { echo "Unknown dataset: ${ds}" >&2; exit 2; }
               case "${TARGET%%_*}" in
                 predict) predict "${ds}" "${DEVICE:-${GPU[$ds]}}" ;;
                 eval)    evaluate "${ds}" ;;
                 diagnostics) diag "${ds}" ;;
                 benchmark) bench "${ds}" "${DEVICE:-${GPU[$ds]}}" ;;
               esac ;;
  help|-h|--help)
    grep '^# ' "${BASH_SOURCE[0]}" | sed 's/^# //' ;;
  *) echo "Unknown target: ${TARGET}" >&2; exit 2 ;;
esac
