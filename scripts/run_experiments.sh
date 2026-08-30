#!/usr/bin/env bash
#
# run_experiments.sh — Wrapper to run all GA experiments and tarball results.
#
# Designed for VPS deployment (Hetzner Cloud). Runs the full GA experiment
# suite via test_genetic_algorithm.py, then tars all results into a single
# archive for easy download via scp.
#
# Logs progress to results/experiment.log so you can check status remotely:
#   ssh root@<vps-ip> "tail -20 /root/gahomotopy_ws/results/experiment.log"
#
# Usage:
#   cd ~/gahomotopy_ws
#   source .venv/bin/activate
#   bash scripts/run_experiments.sh
#
# When done, download the tarball:
#   scp root@<vps-ip>:~/gahomotopy_ws/results.tar.gz ./
#

set -euo pipefail

WORKSPACE_DIR="$(cd "$(dirname "$0")/.." && pwd)"
RESULTS_DIR="${WORKSPACE_DIR}/results"
LOG_FILE="${RESULTS_DIR}/experiment.log"
TARBALL="${WORKSPACE_DIR}/results.tar.gz"

cd "${WORKSPACE_DIR}"

# Ensure results directory exists before writing log
mkdir -p "${RESULTS_DIR}"

echo "============================================" | tee "${LOG_FILE}"
echo "GA Experiment Suite" | tee -a "${LOG_FILE}"
echo "Started:  $(date)" | tee -a "${LOG_FILE}"
echo "Workspace: ${WORKSPACE_DIR}" | tee -a "${LOG_FILE}"
echo "============================================" | tee -a "${LOG_FILE}"

# Pre-flight checks
if [ ! -d ".venv" ]; then
    echo "[ERROR] .venv not found. Run: python3 -m venv .venv && source .venv/bin/activate && pip install -e gahomotopy/" | tee -a "${LOG_FILE}"
    exit 1
fi

# Activate venv
source .venv/bin/activate

# Quick import check
python -c "from gahomotopy.kinematics.roarm import ROARM3DOF; from gahomotopy.planning.genetic_algorithm import GeneticAlgorithm; print('Imports OK')" | tee -a "${LOG_FILE}"

echo "" | tee -a "${LOG_FILE}"
echo "Running experiments..." | tee -a "${LOG_FILE}"
echo "" | tee -a "${LOG_FILE}"

# Run the full experiment suite
START_EPOCH=$(date +%s)
python -m gahomotopy.tests.test_genetic_algorithm 2>&1 | tee -a "${LOG_FILE}"
EXIT_CODE=${PIPESTATUS[0]}
END_EPOCH=$(date +%s)
ELAPSED=$((END_EPOCH - START_EPOCH))

echo "" | tee -a "${LOG_FILE}"
echo "============================================" | tee -a "${LOG_FILE}"
echo "Experiments finished." | tee -a "${LOG_FILE}"
echo "Exit code: ${EXIT_CODE}" | tee -a "${LOG_FILE}"
echo "Elapsed: ${ELAPSED}s ($(( ELAPSED / 3600 ))h $(( (ELAPSED % 3600) / 60 ))m)" | tee -a "${LOG_FILE}"
echo "Finished: $(date)" | tee -a "${LOG_FILE}"
echo "============================================" | tee -a "${LOG_FILE}"

if [ "${EXIT_CODE}" -ne 0 ]; then
    echo "[ERROR] Experiments failed with exit code ${EXIT_CODE}. Not creating tarball." | tee -a "${LOG_FILE}"
    exit ${EXIT_CODE}
fi

# Count output files
FILE_COUNT=$(find "${RESULTS_DIR}" -type f \( -name "*.npy" -o -name "*.json" \) | wc -l)
echo "Output files generated: ${FILE_COUNT}" | tee -a "${LOG_FILE}"

# Create tarball of all results
echo "" | tee -a "${LOG_FILE}"
echo "Creating tarball: ${TARBALL}" | tee -a "${LOG_FILE}"
tar -czf "${TARBALL}" -C "${WORKSPACE_DIR}" results/
TARBALL_SIZE=$(du -h "${TARBALL}" | cut -f1)
echo "Tarball size: ${TARBALL_SIZE}" | tee -a "${LOG_FILE}"

echo "" | tee -a "${LOG_FILE}"
echo "============================================" | tee -a "${LOG_FILE}"
echo "ALL DONE" | tee -a "${LOG_FILE}"
echo "" | tee -a "${LOG_FILE}"
echo "Results tarball: ${TARBALL} (${TARBALL_SIZE})" | tee -a "${LOG_FILE}"
echo "Log file:         ${LOG_FILE}" | tee -a "${LOG_FILE}"
echo "" | tee -a "${LOG_FILE}"
echo "To download from your local machine:" | tee -a "${LOG_FILE}"
echo "  scp root@<vps-ip>:${TARBALL} ./" | tee -a "${LOG_FILE}"
echo "============================================" | tee -a "${LOG_FILE}"