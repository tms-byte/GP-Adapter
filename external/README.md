## LoCoOp / CoOp outputs

Place prompt-learning outputs under `external/locoop_outputs/` if you want to use the default example layout from this repository.
You may also keep them anywhere else and pass the location with `--locoop-output-root`.

Expected structure:

```text
external/locoop_outputs/
  imagenet/
    LoCoOp/
      vit_b16_ep50_16shots/
        nctx16_cscFalse_ctpend/
          seed1/
            prompt_learner/
              checkpoint
              model.pth.tar
    CoOp/
      vit_b16_ep50_16shots/
        nctx16_cscFalse_ctpend/
          seed1/
            prompt_learner/
              checkpoint
              model.pth.tar
  imagenet100/
    LoCoOp/
      vit_b16_ep50_16shots/
        nctx16_cscFalse_ctpend/
          seed1/
            prompt_learner/
              checkpoint
              model.pth.tar
```

Notes:
- `external/locoop_outputs/` mirrors the original LoCoOp output layout. 
- Shared few-shot splits should be stored under `../manifests/` instead of inside prompt-learning output folders.

## LoCoOp configs

Place LoCoOp yaml files under `external/locoop_configs/` if you want to use the default example layout from this repository.
You may also keep them anywhere else and pass the location with `--locoop-config-dir`.

Expected files:

```text
external/locoop_configs/
  vit_b16_ep50.yaml
  rn50_ep50.yaml
```
