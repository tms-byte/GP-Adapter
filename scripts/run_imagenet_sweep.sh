#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT=${DATA_ROOT:-/raid}
OOD_DATA_ROOT=${OOD_DATA_ROOT:-${DATA_ROOT}/ood_images}
LOCOOP_OUTPUT_ROOT=${LOCOOP_OUTPUT_ROOT:-}
LOCOOP_CONFIG_DIR=${LOCOOP_CONFIG_DIR:-}
MANIFEST_ROOT=${MANIFEST_ROOT:-manifests}

SEEDS=(1 2 3)
TRAINERS=("LoCoOp" "CoOp" "NoPrompt")
BACKBONES=("ViT-B/16" "RN50")
DATASETS=("imagenet" "imagenet100")

for dataset in "${DATASETS[@]}"; do
  config_path="datasets/configs/${dataset}.yaml"
  declare -a dataset_flags=("--root-path" "$DATA_ROOT" "--root-path-ood" "$OOD_DATA_ROOT" "--data-dir-name" "ILSVRC2012")
  if [[ "$dataset" == "imagenet100" ]]; then
    dataset_flags=("--imagenet100" "--root-path" "$DATA_ROOT" "--root-path-ood" "$OOD_DATA_ROOT" "--data-dir-name" "ILSVRC2012")
  fi

  for backbone in "${BACKBONES[@]}"; do
    if [[ "$backbone" == "RN50" && "$dataset" == "imagenet100" ]]; then
      echo "[skip] dataset=${dataset} backbone=${backbone} (ablation only)"
      continue
    fi
    backbone_slug=${backbone//\//_}
    for trainer in "${TRAINERS[@]}"; do
      for seed in "${SEEDS[@]}"; do
        log_dir="logs/sweeps/${dataset}/${backbone_slug}/${trainer}/seed${seed}"
        cmd=(
          python main_gpy_imagenet.py
          --config "$config_path"
          --log-dir "$log_dir"
          --backbone "$backbone"
          --seed "$seed"
          --trainer "$trainer"
          --manifest-root "$MANIFEST_ROOT"
          "${dataset_flags[@]}"
        )

        if [[ "$trainer" == "LoCoOp" || "$trainer" == "CoOp" ]]; then
          if [[ -z "$LOCOOP_OUTPUT_ROOT" || -z "$LOCOOP_CONFIG_DIR" ]]; then
            echo "[skip] dataset=${dataset} backbone=${backbone} trainer=${trainer} seed=${seed} (set LOCOOP_OUTPUT_ROOT and LOCOOP_CONFIG_DIR)"
            continue
          fi
          cmd+=(--locoop-output-root "$LOCOOP_OUTPUT_ROOT" --locoop-config-dir "$LOCOOP_CONFIG_DIR")
        fi

        echo "[run] dataset=${dataset} backbone=${backbone} trainer=${trainer} seed=${seed}"
        "${cmd[@]}"
      done
    done
  done
done
