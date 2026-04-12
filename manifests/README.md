# Shared Few-Shot Manifests

This directory stores trainer-independent few-shot split definitions.

Recommended structure:

```text
manifests/
  imagenet/
    vit_b16_ep50_16shots/
      seed1/selected_shots.json
      seed2/selected_shots.json
    rn50_ep50_16shots/
      seed1/selected_shots.json
  imagenet100/
    vit_b16_ep50_16shots/
      seed1/selected_shots.json
```

Notes:
- ImageNet manifests are stored under `manifests/imagenet`, regardless of whether `--data-dir-name` is `imagenet`, `ILSVRC2012`.
- `imagenet100` corresponds to `--data-dir-name imagenet100`.
- You can still override the path directly with `--selected-shots-json`.