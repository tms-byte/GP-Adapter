#!/usr/bin/env python
# coding: utf-8

import json
import logging
import os
import random
import shutil
from collections import OrderedDict
from pathlib import Path

import clip
import gpytorch
import numpy as np
import torch
import torch.nn.functional as F
import torchvision
import yaml
from gpytorch.constraints import Interval
from gpytorch.distributions import MultitaskMultivariateNormal, MultivariateNormal
from gpytorch.likelihoods import GaussianLikelihood
from ood_metrics import calc_metrics

from gp_components import LocalGPImage, LocalGPText, RBFKernelOptimizer
from run_config import (
    CONFIG_ROOT,
    DEFAULT_MANIFEST_ROOT,
    LOGGER_NAME,
    REPO_ROOT,
    build_arg_parser,
    configure_logger,
    create_print_logger,
)
from scipy.stats import norm
from torch.distributions import Normal
from tqdm import tqdm

from datasets.imagenet import ImageNet
from datasets.imagenet100 import ImageNet100
from datasets.imagenet_select import ImageNetSelect
from utils import *

try:
    import pandas as pd
except ImportError:
    pd = None


logger = logging.getLogger(LOGGER_NAME)
def main():
    parser = build_arg_parser()
    cli_args, _ = parser.parse_known_args()
    mix_alpha = cli_args.mix_alpha
    loss_threshold = cli_args.loss_thr
    log_dir_path = Path(cli_args.log_dir)

    cli_imagenet100 = cli_args.imagenet100
    config_path = cli_args.config if cli_args.config else str(CONFIG_ROOT / ("imagenet100.yaml" if cli_imagenet100 else "imagenet.yaml"))

    cfg = yaml.load(open(config_path, 'r'), Loader=yaml.Loader)
    dataset_name = cfg.get('dataset', 'imagenet100' if cli_imagenet100 else 'imagenet')
    is_imagenet100 = cli_imagenet100 or dataset_name == 'imagenet100'
    data_dir_name = cli_args.data_dir_name if cli_args.data_dir_name else ("ILSVRC2012" if is_imagenet100 else "ILSVRC2012")

    if cli_args.root_path:
        cfg['root_path'] = cli_args.root_path
    if cli_args.root_path_ood:
        cfg['root_path_ood'] = cli_args.root_path_ood
    if cli_args.backbone:
        cfg['backbone'] = cli_args.backbone
    if cli_args.shots is not None:
        cfg['shots'] = cli_args.shots

    run_tag = "_".join([
        f"seed{cli_args.seed}",
        cli_args.trainer.lower(),
        dataset_name,
        cfg['backbone'].replace('/', '').replace('-', ''),
        f"mix{str(mix_alpha).replace('.', 'p')}",
        f"loss{str(loss_threshold).replace('-', 'm').replace('.', 'p')}"
    ])

    log_path = configure_logger(logger, cli_args.log_file, cli_args.log_dir, config_path, run_tag=run_tag)
    print = create_print_logger(logger)
    logger.info("Logging to %s", log_path)


    cache_dir = os.path.join('./caches', dataset_name)
    os.makedirs(cache_dir, exist_ok=True)
    cfg['cache_dir'] = cache_dir

    print("\nRunning configs.")
    print(cfg, "\n")
    print(f"Hyperparameters -> mix_alpha: {mix_alpha}, loss_threshold: {loss_threshold}")

    # CLIP
    clip_model, preprocess = clip.load(cfg['backbone'])
    clip_model.eval()

    # Prepare dataset
    SEED = cli_args.seed
    TRAINER = cli_args.trainer
    if cfg['backbone'] == "RN50":
        backbone_cfg = "rn50_ep50"
    elif cfg['backbone'] == "ViT-B/16":
        backbone_cfg = "vit_b16_ep50"

    random.seed(SEED)
    torch.manual_seed(SEED)

    print("Preparing dataset.")

    shots_tag = f"{cfg['shots']}shots"
    shots_trainer = "LoCoOp"
    locoop_output_root = Path(cli_args.locoop_output_root).expanduser() if cli_args.locoop_output_root else None
    locoop_config_dir = Path(cli_args.locoop_config_dir).expanduser() if cli_args.locoop_config_dir else None
    manifest_root = Path(cli_args.manifest_root).expanduser() if cli_args.manifest_root else DEFAULT_MANIFEST_ROOT
    manifest_data_dir = dataset_name
    
    if manifest_data_dir.lower() in {"imagenet", "ilsvrc2012", "islvrc2012"}:
        manifest_data_dir = "imagenet"
    
    shots_base = None
    selected_json_path = Path(cli_args.selected_shots_json).expanduser() if cli_args.selected_shots_json else None
    if selected_json_path is None and manifest_root is not None:
        manifest_candidate = manifest_root / manifest_data_dir / f"{backbone_cfg}_{shots_tag}" / f"seed{SEED}" / "selected_shots.json"
        if manifest_candidate.is_file():
            selected_json_path = manifest_candidate
    
    if locoop_output_root is not None:
        shots_base = locoop_output_root / dataset_name / shots_trainer / f"{backbone_cfg}_{shots_tag}" / "nctx16_cscFalse_ctpend" / f"seed{SEED}"
        if selected_json_path is None:
            legacy_candidate = shots_base / "selected_shots.json"
            if legacy_candidate.is_file():
                selected_json_path = legacy_candidate
    
    prompt_trainer = TRAINER if TRAINER in ("LoCoOp", "CoOp") else None
    prompt_ckpt_path = None
    if prompt_trainer:
        if locoop_output_root is None:
            raise ValueError("--locoop-output-root or LOCOOP_OUTPUT_ROOT is required when trainer is LoCoOp or CoOp.")
        prompt_ckpt_path = locoop_output_root / dataset_name / prompt_trainer / f"{backbone_cfg}_{shots_tag}" / "nctx16_cscFalse_ctpend" / f"seed{SEED}"
    trainer_cfg_path = (locoop_config_dir / f"{backbone_cfg}.yaml") if locoop_config_dir is not None else None
    
    if prompt_trainer and trainer_cfg_path is None:
        raise ValueError("--locoop-config-dir or LOCOOP_CONFIG_DIR is required when trainer is LoCoOp or CoOp.")
    use_selected_shots = selected_json_path is not None and selected_json_path.is_file()
    
    if prompt_ckpt_path is not None:
        logger.info("Prompt checkpoint path: %s", prompt_ckpt_path)
    if selected_json_path is not None:
        logger.info("Selected shots manifest path: %s", selected_json_path)

    if is_imagenet100:
        if use_selected_shots:
            imagenet = ImageNet100(
                cfg['root_path'],
                cfg['shots'],
                preprocess,
                selected_json=str(selected_json_path),
                strict=False,
                dataset_dir=data_dir_name,
            )
        else:
            imagenet = ImageNet100(cfg['root_path'], cfg['shots'], preprocess, dataset_dir=data_dir_name)
    else:
        if use_selected_shots:
            imagenet = ImageNetSelect(
                cfg['root_path'],
                str(selected_json_path),
                preprocess,
                strict=False,
                dataset_dir=data_dir_name,
            )
        else:
            imagenet = ImageNet(cfg['root_path'], cfg['shots'], preprocess, dataset_dir=data_dir_name)

    test_loader = torch.utils.data.DataLoader(imagenet.test, batch_size=256, num_workers=8, shuffle=False)
    train_loader_cache = torch.utils.data.DataLoader(imagenet.train, batch_size=16, num_workers=8, shuffle=False)

    # Construct the cache model by few-shot training set
    # cache_keys:image_features, cache_values:encoded_labels
    print("\nConstructing cache model by few-shot visual features and labels.")
    cache_keys, cache_values = build_cache_model(cfg, clip_model, train_loader_cache)
    print("cache_keys:",cache_keys.shape)
    cache_values = cache_values.view(len(imagenet.classnames),cfg['shots'],-1)
    print("cache_values:",cache_values.shape)

    # Textual features
    print("\nGetting textual features as CLIP's classifier.")
    trainer_mode = TRAINER.lower()
    use_prompt_ckpt = trainer_mode in ("locoop", "coop")
    
    if use_prompt_ckpt:
        clip_weights = clip_classifier_locoop(
            imagenet.classnames,
            template=imagenet.template,
            clip_model=clip_model,
            sigma=0.0,
            prompt_ckpt_path=str(prompt_ckpt_path),
            cfg_path=str(trainer_cfg_path),
            use_gaussian_sampling=False
        )
        print(clip_weights.shape)
        clip_weights = clip_weights.reshape(-1,clip_weights.size(-1)).transpose(1,0)
    else:
        clip_weights = clip_classifier(imagenet.classnames, imagenet.template, clip_model)
    print("clip_weights:",clip_weights.shape)

    shots = cfg['shots']
    num_classes = clip_weights.size(1)
    text_feats = clip_weights.t().contiguous()
    image_feats = cache_keys.t().contiguous()

    num_classes = len(imagenet.classnames)

    image_subset = (image_feats.view(num_classes, shots, -1))    # [1000, 16, 512]
    
    X_img=image_subset.to(dtype=torch.float32)  # [1000, 16, 512]
    X_text = text_feats.unsqueeze(1)    # [1000, 1, 512]
    
    y=cache_values
    y = torch.argmax(y,axis=-1)

    num_classes=len(imagenet.classnames)
    y_img = torch.ones_like(y,dtype=torch.float32)
    y_text = torch.ones_like(y[:,:1],dtype=torch.float32)

    # compute linear kernel value for text kernel
    kernel = gpytorch.kernels.LinearKernel()
    phi = kernel(X_text.cpu(), X_text.cpu()).evaluate()  # φ = X X^T
    phi_inv = torch.linalg.inv(phi + 1e-3 * torch.eye(phi.size(-1)))
    numerator = torch.bmm(torch.bmm(y_text.unsqueeze(1).cpu(), phi_inv), y_text.unsqueeze(2).cpu()).squeeze()
    n = shots
    tau2_mle_batch_text = numerator / n  # (1000,)
    print(f"MLE estimate of tau^2_text: {tau2_mle_batch_text[:10]}")

    optimizer = RBFKernelOptimizer(X_img, y_img, loss_thr=loss_threshold)
    results = optimizer.compute_classwise_optima()
    tau2_mle_batch_img = results["best_tau2"]

    for i in range(5):
        print(f"Class {i}: length-scale={results['best_length_scale'][i]:.2f}, tau^2={results['best_tau2'][i]:.4f}, log likelihood={results['best_log_likelihood'][i]:.2f}")

    likelihood_img = GaussianLikelihood(batch_shape=torch.Size([num_classes],noise_constraint=Interval(1e-6, 1e-5)))
    likelihood_img.noise = 1.0e-6
    likelihood_txt = GaussianLikelihood(batch_shape=torch.Size([num_classes],noise_constraint=Interval(1e-6, 1e-5)))
    likelihood_txt.noise = 1.0e-6
    model = LocalGPImage(X_img, y_img, likelihood=likelihood_img, num_classes=num_classes, tau2_mle=tau2_mle_batch_img)
    model.covar_module1.lengthscale = results["best_length_scale"].view(len(imagenet.classnames), 1, 1)
    model2 = LocalGPText(X_text, y_text, likelihood=likelihood_txt, num_classes=num_classes, tau2_mle=tau2_mle_batch_text)


    model.cuda().eval()
    model2.cuda().eval()
    likelihood_img.cuda().eval()
    likelihood_txt.cuda().eval()

    normalized=False

    num_sample = 64

    correct = 0
    correct_probit = 0
    total = 0

    pred_list = None
    epi_list = None
    pred_list_probit = None
    msp_var_list = None

    temperature=1.0
    bias = 0.0
    a = mix_alpha

    with torch.no_grad(), gpytorch.settings.num_likelihood_samples(num_sample):
        for i, (images, target) in enumerate(tqdm(test_loader)):
            images, target = images.cuda(), target.cuda()
            image_features = clip_model.encode_image(images)
            image_features = torch.tensor(image_features,dtype=torch.float32)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            features_share = image_features.unsqueeze(0).expand(num_classes,-1,-1)
    
            output = model(features_share)
            output2 = model2(features_share)
            mean = output.mean*a+(1-a)*output2.mean
            variance =torch.diagonal(output.covariance_matrix*(a**2) + output2.covariance_matrix*((1-a)**2),dim1=-2,dim2=-1)
            scaled_variance = (variance / temperature) - bias# + 1e-3 * torch.eye(variance.size(-1), device=variance.device) 
        
            adjusted_mean = mean / torch.sqrt(1 + (3 * scaled_variance) / (np.pi**2))
            probit_probs = torch.tensor(adjusted_mean.cpu().detach().numpy(), device=mean.device)
            pred_samples = probit_probs.exp()#output.sample(torch.Size((num_sample,))).exp()
            probabilities_probit = (pred_samples / pred_samples.sum(0))
            pred_samples = mean.exp()
            probabilities = (pred_samples / pred_samples.sum(0))
            msp_var = torch.max(probabilities,0)[0] + (torch.max(probabilities,0)[0]/(scaled_variance.max(0)[0]+scaled_variance.min(0)[0]))*(scaled_variance.max(0)[0]-scaled_variance.min(0)[0])

            if normalized:
                probabilities_probit = F.softplus(probabilities_probit)
            predicted_probit = torch.argmax(probabilities_probit, dim=0)
            predicted = torch.argmax(probabilities, dim=0)

            try:
                pred_list = torch.vstack((pred_list,probabilities.permute(1,0)))
                # epi_list = torch.vstack((epi_list,torch.diagonal(variance, dim1=-2, dim2=-1).permute(1,0)))
                epi_list = torch.vstack((epi_list,variance.permute(1,0)))
                pred_list_probit = torch.vstack((pred_list_probit,probabilities_probit.permute(1,0)))
                msp_var_list = torch.cat([msp_var_list,msp_var],dim=0)
            except:
                pred_list = probabilities.permute(1,0)
                # epi_list = torch.diagonal(variance, dim1=-2, dim2=-1).permute(1,0)
                epi_list = variance.permute(1,0)
                pred_list_probit = probabilities_probit.permute(1,0)
                msp_var_list = msp_var
        
            correct += (predicted == target).sum().item()
            correct_probit += (predicted_probit == target).sum().item()
            total += target.size(0)


    print(f"epistemic: {epi_list.mean(dim=0).sum()}")
    accuracy = correct / total
    accuracy_probit = correct_probit / total

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Accuracy_probit: {accuracy_probit:.4f}")


    model.state_dict().keys()

    def set_ood_loader(out_dataset, preprocess, root, batch_size):
        """
        set ood loader for ImageNet scale dataset\n
        https://github.com/deeplearning-wisc/MCM/blob/main/utils/detection_util.py#L12
        """
        dataset_key = out_dataset.lower()
        if dataset_key == 'inaturalist':
            testsetout = torchvision.datasets.ImageFolder(root=os.path.join(root, 'iNaturalist'), transform=preprocess)
        elif dataset_key == 'sun':
            testsetout = torchvision.datasets.ImageFolder(root=os.path.join(root, 'SUN'), transform=preprocess)
        elif dataset_key == 'places':
            testsetout = torchvision.datasets.ImageFolder(root=os.path.join(root, 'Places'), transform=preprocess)
        elif dataset_key == 'placesbg':
            testsetout = torchvision.datasets.ImageFolder(root=os.path.join(root, 'placesbg'), transform=preprocess)
        elif dataset_key == 'dtd':
            testsetout = torchvision.datasets.ImageFolder(root=os.path.join(root, 'dtd'), transform=preprocess)
        else:
            raise ValueError(f'Unsupported OOD dataset: {out_dataset}')

        testloaderOut = torch.utils.data.DataLoader(
            testsetout,
            batch_size=batch_size,
            shuffle=False,
            num_workers=4,
        )
        return testloaderOut


    OOD_DATASETS = ["iNaturalist", "SUN", "Places", "DTD"]


    def build_ood_cfg(dataset_name):
        cfg_ood = {
            "dataset": dataset_name,
            "root_path": cfg.get("root_path_ood", cfg.get("root_path")),
            "load_pre_feat": cfg.get("load_pre_feat", False),
            "backbone": cfg["backbone"],
        }
        cache_dir_ood = os.path.join('./caches', cfg_ood['dataset'])
        os.makedirs(cache_dir_ood, exist_ok=True)
        cfg_ood['cache_dir'] = cache_dir_ood
        return cfg_ood


    cfg_ood = build_ood_cfg("iNaturalist")

    print("\nRunning configs.")
    print(cfg_ood, "\n")

    dataset_outlier = set_ood_loader(cfg_ood['dataset'], preprocess, cfg_ood['root_path'], 512)
    # Pre-load test features
    print("\nLoading visual features and labels from test set.")
    test_features_outlier, test_labels_outlier = pre_load_features(cfg_ood, "test", clip_model, dataset_outlier)


    pred_list_outlier = None
    epi_list_outlier = None
    pred_list_probit_outlier = None
    msp_var_list_outlier = None

    short_cut = test_features_outlier.size(0)
    with torch.no_grad(), gpytorch.settings.num_likelihood_samples(num_sample):
        for ite in tqdm(range(0,short_cut,64)):
            features = test_features_outlier[ite:ite+64]
            # features = torch.hstack((feat,feat))
            features_share = features.unsqueeze(0).expand(num_classes,-1,-1)

            output = model(features_share)
            output2 = model2(features_share)
            mean = output.mean*a+(1-a)*output2.mean
            variance = torch.diagonal(output.covariance_matrix*(a**2) + output2.covariance_matrix*((1-a)**2),dim1=-2,dim2=-1)
                    
            scaled_variance = (variance-bias) / temperature #+ 1e-3 * torch.eye(variance.size(-1), device=variance.device)
        
            adjusted_mean = mean / torch.sqrt(1 + (3 * scaled_variance) / (np.pi**2))
            probit_probs = torch.tensor(adjusted_mean.cpu().detach().numpy(), device=mean.device)
        
            pred_samples = probit_probs.exp()#output.sample(torch.Size((num_sample,))).exp()
            probabilities_probit = (pred_samples / pred_samples.sum(0))
            if normalized:
                probabilities_probit = F.softplus(probabilities_probit)
            predicted = torch.argmax(probabilities_probit, dim=0)

            pred_samples = mean.exp()#output.sample(torch.Size((num_sample,))).exp()
            probabilities = (pred_samples / pred_samples.sum(0))
            predicted = torch.argmax(probabilities, dim=0)
            msp_var = torch.max(probabilities,0)[0] + (torch.max(probabilities,0)[0]/(scaled_variance.max(0)[0]+scaled_variance.min(0)[0]))*(scaled_variance.max(0)[0]-scaled_variance.min(0)[0])

        
            try:
                pred_list_outlier = torch.vstack((pred_list_outlier,probabilities.permute(1,0)))
                epi_list_outlier = torch.vstack((epi_list_outlier,variance.permute(1,0)))
                pred_list_probit_outlier = torch.vstack((pred_list_probit_outlier,probabilities_probit.permute(1,0)))
                msp_var_list_outlier = torch.cat([msp_var_list_outlier,msp_var],dim=0)
            except:
                pred_list_outlier = probabilities.permute(1,0)
                epi_list_outlier = variance.permute(1,0)
                pred_list_probit_outlier = probabilities_probit.permute(1,0)
                msp_var_list_outlier = msp_var

        print(pred_list_outlier.size(),test_labels_outlier[:short_cut].size())
        print(f"epistemic: {epi_list_outlier.mean(dim=0).sum()}")



    from ood_metrics import calc_metrics

    top_k=1
    preds = torch.cat((pred_list_probit.topk(top_k, dim=1, largest=True)[0].mean(dim=1),pred_list_probit_outlier.topk(top_k, dim=1, largest=True)[0].mean(dim=1)))
    pred_label = torch.ones(pred_list.size(0))
    pred_label_outlier = torch.zeros(pred_list_probit_outlier.size(0))
    pred_label = torch.cat((pred_label,pred_label_outlier))
    print("maxp_probit")
    print(calc_metrics(preds.detach().cpu().numpy(),pred_label),preds.shape)

    print("\nentropy_probit")
    preds_entropy = torch.cat((torch.sum(-pred_list_probit*(pred_list_probit+1e-6).log(),dim=1),torch.sum(-pred_list_probit_outlier*(pred_list_probit_outlier+1e-6).log(),dim=1)))
    print(calc_metrics(4-preds_entropy.detach().cpu().numpy(),pred_label),preds.shape)

    preds = torch.cat((pred_list.max(dim=1)[0],pred_list_outlier.max(dim=1)[0]))
    pred_label = torch.ones(pred_list.size(0))
    pred_label_outlier = torch.zeros(pred_list_outlier.size(0))
    pred_label = torch.cat((pred_label,pred_label_outlier))
    print("\nmaxp")
    print(calc_metrics(preds.detach().cpu().numpy(),pred_label),preds.shape)

    print("\nentropy")
    preds_entropy = torch.cat((torch.sum(-pred_list*(pred_list+1e-3).log(),dim=1),torch.sum(-pred_list_outlier*(pred_list_outlier+1e-3).log(),dim=1)))
    print(calc_metrics(4-preds_entropy.detach().cpu().numpy(),pred_label),preds.shape)

    print("\nepistemic")
    preds_epi = torch.cat((epi_list.mean(dim=1),epi_list_outlier.mean(dim=1)))
    print(calc_metrics(1-preds_epi.detach().cpu().numpy(),pred_label),preds_epi.shape)

    print("\nvariance_max-min")
    preds_epi = torch.cat((epi_list.min(dim=1)[0]-epi_list.max(dim=1)[0],epi_list_outlier.min(dim=1)[0]-epi_list_outlier.max(dim=1)[0]))
    print(calc_metrics(1-preds_epi.detach().cpu().numpy(),pred_label),preds_epi.shape)

    max_indices = torch.argmax(pred_list_probit, dim=1)
    max_indices_outlier = torch.argmax(pred_list_probit_outlier, dim=1)
    epi_list_sel = epi_list[torch.arange(epi_list.size(0)), max_indices]
    epi_list_sel_outlier = epi_list_outlier[torch.arange(epi_list_outlier.size(0)), max_indices_outlier]

    print("\nvariance_sel")
    preds_epi = torch.cat((epi_list_sel,epi_list_sel_outlier))
    print(calc_metrics(1-preds_epi.detach().cpu().numpy(),pred_label),preds_epi.shape)

    print("\nvariance_min")
    preds_epi_p = torch.cat((epi_list.min(dim=1)[0],epi_list_outlier.min(dim=1)[0]))
    print(calc_metrics(1-preds_epi_p.detach().cpu().numpy(),pred_label),preds_epi_p.shape)

    print("\nmsp_var")
    preds_msp_var = torch.cat((msp_var_list,msp_var_list_outlier))
    print(calc_metrics(preds_msp_var.detach().cpu().numpy(),pred_label),preds_msp_var.shape)

    def evaluate_ood_dataset(
        dataset_name,
        dataset_label=None,
        update_global_state=False,
    ):
        """Run the full OOD evaluation pipeline for one dataset and return metrics."""
        cfg_ood_local = build_ood_cfg(dataset_name)
        dataset_label = dataset_label or cfg_ood_local['dataset']

        dataset_outlier = set_ood_loader(cfg_ood_local['dataset'], preprocess, cfg_ood_local['root_path'], 512)
        test_features_outlier, test_labels_outlier = pre_load_features(cfg_ood_local, "test", clip_model, dataset_outlier)

        pred_list_outlier = None
        epi_list_outlier = None
        pred_list_probit_outlier = None
        msp_var_list_outlier = None

        short_cut = test_features_outlier.size(0)
        with torch.no_grad(), gpytorch.settings.num_likelihood_samples(num_sample):
            for ite in tqdm(range(0, short_cut, 64)):
                features = test_features_outlier[ite:ite+64]
                features_share = features.unsqueeze(0).expand(num_classes, -1, -1)

                output = model(features_share)
                output2 = model2(features_share)
                mean = output.mean * a + (1 - a) * output2.mean
                variance = torch.diagonal(
                    output.covariance_matrix * (a ** 2) + output2.covariance_matrix * ((1 - a) ** 2),
                    dim1=-2,
                    dim2=-1,
                )

                scaled_variance = (variance - bias) / temperature
                adjusted_mean = mean / torch.sqrt(1 + (3 * scaled_variance) / (np.pi ** 2))
                probit_probs = torch.tensor(adjusted_mean.cpu().detach().numpy(), device=mean.device)
                pred_samples_probit = probit_probs.exp()
                probabilities_probit = pred_samples_probit / pred_samples_probit.sum(0)
                if normalized:
                    probabilities_probit = F.softplus(probabilities_probit)

                pred_samples = mean.exp()
                probabilities = pred_samples / pred_samples.sum(0)
                msp_var = torch.max(probabilities,0)[0] + (torch.max(probabilities,0)[0]/(scaled_variance.max(0)[0]+scaled_variance.min(0)[0]))*(scaled_variance.max(0)[0]-scaled_variance.min(0)[0])
                
                try:
                    pred_list_outlier = torch.vstack((pred_list_outlier, probabilities.permute(1, 0)))
                    epi_list_outlier = torch.vstack((epi_list_outlier, variance.permute(1, 0)))
                    pred_list_probit_outlier = torch.vstack((pred_list_probit_outlier, probabilities_probit.permute(1, 0)))
                    msp_var_list_outlier = torch.cat([msp_var_list_outlier, msp_var], dim=0)
                except Exception:
                    pred_list_outlier = probabilities.permute(1, 0)
                    epi_list_outlier = variance.permute(1, 0)
                    pred_list_probit_outlier = probabilities_probit.permute(1, 0)
                    msp_var_list_outlier = msp_var

        max_indices = torch.argmax(pred_list_probit, dim=1)
        max_indices_outlier = torch.argmax(pred_list_probit_outlier, dim=1)
        epi_list_sel = epi_list[torch.arange(epi_list.size(0)), max_indices]
        epi_list_sel_outlier = epi_list_outlier[torch.arange(epi_list_outlier.size(0)), max_indices_outlier]

        metrics_by_method = OrderedDict()

        def _add_metric(name, in_tensor, out_tensor):
            preds_cat = torch.cat((in_tensor, out_tensor))
            labels = torch.cat(
                (
                    torch.ones(in_tensor.size(0), device=in_tensor.device),
                    torch.zeros(out_tensor.size(0), device=out_tensor.device),
                )
            )
            metrics_by_method[name] = calc_metrics(
                preds_cat.detach().cpu().numpy(),
                labels.detach().cpu().numpy(),
            )

        top_k = 1
        _add_metric(
            "maxp_probit",
            pred_list_probit.topk(top_k, dim=1, largest=True)[0].mean(dim=1),
            pred_list_probit_outlier.topk(top_k, dim=1, largest=True)[0].mean(dim=1),
        )

        entropy_in = torch.sum(-pred_list_probit * (pred_list_probit + 1e-6).log(), dim=1)
        entropy_out = torch.sum(-pred_list_probit_outlier * (pred_list_probit_outlier + 1e-6).log(), dim=1)
        _add_metric("entropy_probit", 4 - entropy_in, 4 - entropy_out)

        _add_metric("maxp", pred_list.max(dim=1)[0], pred_list_outlier.max(dim=1)[0])

        entropy_in_plain = torch.sum(-pred_list * (pred_list + 1e-3).log(), dim=1)
        entropy_out_plain = torch.sum(-pred_list_outlier * (pred_list_outlier + 1e-3).log(), dim=1)
        _add_metric("entropy", 4 - entropy_in_plain, 4 - entropy_out_plain)

        epi_mean_in = epi_list.mean(dim=1)
        epi_mean_out = epi_list_outlier.mean(dim=1)
        _add_metric("epistemic", 1 - epi_mean_in, 1 - epi_mean_out)

        variance_range_in = epi_list.min(dim=1)[0] - epi_list.max(dim=1)[0]
        variance_range_out = epi_list_outlier.min(dim=1)[0] - epi_list_outlier.max(dim=1)[0]
        _add_metric("variance_max-min", 1 - variance_range_in, 1 - variance_range_out)

        _add_metric("variance_sel", 1 - epi_list_sel, 1 - epi_list_sel_outlier)

        variance_min_in = epi_list.min(dim=1)[0]
        variance_min_out = epi_list_outlier.min(dim=1)[0]
        _add_metric("variance_min", 1 - variance_min_in, 1 - variance_min_out)

        preds_msp_var = torch.cat((msp_var_list, msp_var_list_outlier))
        labels_msp = torch.cat(
            (
                torch.ones(msp_var_list.size(0), device=msp_var_list.device),
                torch.zeros(msp_var_list_outlier.size(0), device=msp_var_list_outlier.device),
            )
        )
        metrics_by_method["msp_var"] = calc_metrics(
            preds_msp_var.detach().cpu().numpy(),
            labels_msp.detach().cpu().numpy(),
        )


        if update_global_state:
            globals().update(
                {
                    'cfg_ood': cfg_ood_local,
                    'dataset_outlier': dataset_outlier,
                    'test_features_outlier': test_features_outlier,
                    'test_labels_outlier': test_labels_outlier,
                    'pred_list_outlier': pred_list_outlier,
                    'epi_list_outlier': epi_list_outlier,
                    'pred_list_probit_outlier': pred_list_probit_outlier,
                    'msp_var_list_outlier': msp_var_list_outlier,
                }
            )

        return {
            'dataset': dataset_label,
            'cfg': cfg_ood_local,
            'metrics': metrics_by_method,
        }



    all_ood_metrics = OrderedDict()
    config_items = [(name, name) for name in OOD_DATASETS]
    for idx, (dataset_label, config_path) in enumerate(config_items):
        print(f"Running OOD evaluation for {dataset_label}")
        result = evaluate_ood_dataset(
            dataset_label,
            dataset_label=dataset_label,
            update_global_state=(idx == len(config_items) - 1),
        )
        all_ood_metrics[dataset_label] = result['metrics']
        for metric_name, stats in result['metrics'].items():
            print(f"  {metric_name}: {stats}")
        print()

    rows = []
    for dataset_label, metrics in all_ood_metrics.items():
        for metric_name, stats in metrics.items():
            row = {'dataset': dataset_label, 'score': metric_name}
            row.update(stats)
            rows.append(row)


    def _prepare_ood_log_base_path():
        log_dir = os.path.join(
            str(REPO_ROOT),
            'logs',
            f'seed{SEED}',
            'prompt_learner',
        )
        os.makedirs(log_dir, exist_ok=True)
        trainer_lower = TRAINER.lower()
        trainer_slug = {
            'locoop': 'locoop',
            'coop': 'coop',
            'noprompt': 'noprompt'
        }.get(trainer_lower, trainer_lower)
        backbone_name = cfg.get('backbone', '')
        backbone_lower = backbone_name.lower()
        if 'rn50' in backbone_lower:
            backbone_slug = 'rn50'
        elif 'vit' in backbone_lower and '16' in backbone_lower:
            backbone_slug = 'vit16'
        else:
            backbone_slug = backbone_lower.replace('/', '').replace('-', '')
        dataset_slug = dataset_name.lower()
        shots_slug = f"shots{cfg.get('shots', 'na')}"
        mix_slug = f"mix{str(mix_alpha).replace('.', 'p')}"
        loss_slug = f"loss{str(loss_threshold).replace('-', 'm').replace('.', 'p')}"
        return os.path.join(log_dir, f'ood_metrics_{trainer_slug}_{backbone_slug}_{dataset_slug}_{shots_slug}_{mix_slug}_{loss_slug}')


    if pd is not None and rows:
        ood_metrics_df = pd.DataFrame(rows)
        print("OOD metrics table:")
        print(ood_metrics_df.to_string(index=False))
        log_base = _prepare_ood_log_base_path()
        csv_path = f"{log_base}.csv"
        ood_metrics_df.to_csv(csv_path, index=False)
        print(f"OOD metrics table saved to {csv_path}")
        log_dir_path.mkdir(parents=True, exist_ok=True)
        csv_log_copy = log_dir_path / Path(csv_path).name
        shutil.copy(csv_path, csv_log_copy)
        print(f"OOD metrics table copied to log dir: {csv_log_copy}")
        try:
            summary = ood_metrics_df.pivot_table(index='score', columns='dataset', values='auroc')
            print("AUROC summary (score x dataset):")
            print(summary.to_string())
        except Exception:
            pass
        metric_cols = {'fpr_at_95_tpr', 'auroc'}
        if metric_cols.issubset(ood_metrics_df.columns):
            mean_metrics = (
                ood_metrics_df
                .groupby('score')[['fpr_at_95_tpr', 'auroc']]
                .mean()
                .reset_index()
                .rename(columns={
                    'fpr_at_95_tpr': 'fpr_at_95_tpr_mean',
                    'auroc': 'auroc_mean'
                })
            )
            print("Score-wise mean metrics:")
            print(mean_metrics.to_string(index=False))
            mean_csv_path = f"{log_base}_score_means.csv"
            mean_metrics.to_csv(mean_csv_path, index=False)
            print(f"Score-wise mean metrics saved to {mean_csv_path}")
            mean_log_copy = log_dir_path / Path(mean_csv_path).name
            shutil.copy(mean_csv_path, mean_log_copy)
            print(f"Score-wise mean metrics copied to log dir: {mean_log_copy}")
    else:
        ood_metrics_df = rows
        print('pandas is not available; metrics stored as a list of dicts.')
        if rows:
            log_base = _prepare_ood_log_base_path()
            json_path = f"{log_base}.json"
            with open(json_path, 'w') as fp:
                json.dump(rows, fp, indent=2)
            print(f"OOD metrics list saved to {json_path}")


if __name__ == "__main__":
    main()
