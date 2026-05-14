#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
V0_DIR="$(cd "${SCRIPT_DIR}" && pwd)"
RAW_DIR="${V0_DIR}/data_clean/raw_clean"
PROCESSED_DIR="${V0_DIR}/data_clean/processed_clean"
DATA_NOTE="${V0_DIR}/data_clean/data_clean.txt"
ZIP_PATH="${RAW_DIR}/game-recommendations-on-steam.zip"
KAGGLE_URL="https://www.kaggle.com/api/v1/datasets/download/antonkozyriev/game-recommendations-on-steam"

mkdir -p "${RAW_DIR}" "${PROCESSED_DIR}"
printf "%s\n" \
  "Ignoring the data files due to git push file size limits." \
  "Also, it is better not to upload this data as it can be reproduced." \
  "Has only two folders -- raw_clean and processed_clean." \
  "raw_clean/ has raw input data files and the zip file for the stabilized v0 MVP." \
  "processed_clean/ has all processed datasets for the cleaned v0 MVP -- item vectors, sampled interactions, temporal train-test split files and computed user profiles." \
  > "${DATA_NOTE}"
curl -L -o "${ZIP_PATH}" "${KAGGLE_URL}"
unzip -o "${ZIP_PATH}" -d "${RAW_DIR}"
rm -rf SCRIPT_DIR/data_clean
