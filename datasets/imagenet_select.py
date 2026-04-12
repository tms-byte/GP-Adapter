import json
import os
from typing import Dict, List, Tuple

from torchvision.datasets import ImageFolder

from .imagenet import (
    imagenet_classes,
    imagenet_templates,
)


class ImageNetSelect:
    """ImageNet dataset whose training split is defined by a JSON manifest.

    The JSON file must contain ``per_class`` mapping WordNet IDs (folder names)
    to the list of image paths that should be used for training. Each entry can
    be either an absolute path or a path relative to ``root/<dataset_dir>``.
    """

    dataset_dir = "ILSVRC2012"

    def __init__(self, root: str, selected_json: str, preprocess, strict: bool = True, dataset_dir: str = None):
        base_dir = dataset_dir or self.dataset_dir
        self.dataset_dir = os.path.join(root, base_dir)
        self.image_dir = self.dataset_dir

        self.train = ImageFolder(root=os.path.join(self.image_dir, "train"), transform=preprocess)
        self.val = ImageFolder(root=os.path.join(self.image_dir, "val"), transform=preprocess)
        self.test = ImageFolder(root=os.path.join(self.image_dir, "val"), transform=preprocess)

        self.template = imagenet_templates
        self.classnames = imagenet_classes

        selected_json = os.path.abspath(os.path.expanduser(selected_json))
        manifest = self._load_manifest(selected_json)
        imgs, targets = self._gather_selected_samples(manifest["per_class"], strict=strict)
        if not imgs:
            raise RuntimeError(
                f"No valid training samples were collected from {selected_json}. "
                "Set strict=False to skip missing files if necessary."
            )

        self.manifest = manifest
        self.selected_json = selected_json
        self.train.samples = imgs
        self.train.imgs = imgs
        self.train.targets = targets

    def _load_manifest(self, selected_json: str) -> Dict:
        if not os.path.isfile(selected_json):
            raise FileNotFoundError(f"Selected-shot manifest not found: {selected_json}")
        with open(selected_json, "r", encoding="utf-8") as f:
            payload = json.load(f)
        if "per_class" not in payload:
            raise ValueError(f"Manifest {selected_json} does not contain 'per_class' entries.")
        return payload

    def _gather_selected_samples(
        self, per_class: Dict[str, List[str]], strict: bool
    ) -> Tuple[List[Tuple[str, int]], List[int]]:
        samples: List[Tuple[str, int]] = []
        targets: List[int] = []

        missing_classes: List[str] = []
        missing_files: List[str] = []

        for wnid, paths in per_class.items():
            if wnid not in self.train.class_to_idx:
                missing_classes.append(wnid)
                continue

            label = self.train.class_to_idx[wnid]
            for path in paths:
                resolved = self._resolve_path(path, wnid)
                if not os.path.exists(resolved):
                    missing_files.append(resolved)
                    if strict:
                        raise FileNotFoundError(
                            f"File listed in manifest was not found: {resolved}"
                        )
                    continue
                samples.append((resolved, label))
                targets.append(label)

        if missing_classes and strict:
            raise ValueError(
                "The following classes were not found in the ImageNet training folder: "
                + ", ".join(sorted(missing_classes))
            )

        if missing_files and not strict:
            print(
                "[ImageNetSelect] Warning: skipped {} entries missing on disk.".format(
                    len(missing_files)
                )
            )

        return samples, targets

    def _resolve_path(self, path: str, wnid: str) -> str:
        if os.path.isabs(path):
            return path

        candidate = os.path.join(self.dataset_dir, path.lstrip(os.sep))
        if os.path.exists(candidate):
            return candidate

        return os.path.join(self.dataset_dir, "train", wnid, os.path.basename(path))
