import json
import os
import random
from collections import defaultdict
from typing import Dict, Iterable, List, Tuple

from torchvision.datasets import ImageFolder

from .imagenet import imagenet_templates


IMAGENET100_WNIDS: List[str] = [
    "n03877845", "n03000684", "n03110669", "n03710721", "n02825657", "n02113186",
    "n01817953", "n04239074", "n02002556", "n04356056", "n03187595", "n03355925",
    "n03125729", "n02058221", "n01580077", "n03016953", "n02843684", "n04371430",
    "n01944390", "n03887697", "n04037443", "n02493793", "n01518878", "n03840681",
    "n04179913", "n01871265", "n03866082", "n03180011", "n01910747", "n03388549",
    "n03908714", "n01855032", "n02134084", "n03400231", "n04483307", "n03721384",
    "n02033041", "n01775062", "n02808304", "n13052670", "n01601694", "n04136333",
    "n03272562", "n03895866", "n03995372", "n06785654", "n02111889", "n03447721",
    "n03666591", "n04376876", "n03929855", "n02128757", "n02326432", "n07614500",
    "n01695060", "n02484975", "n02105412", "n04090263", "n03127925", "n04550184",
    "n04606251", "n02488702", "n03404251", "n03633091", "n02091635", "n03457902",
    "n02233338", "n02483362", "n04461696", "n02871525", "n01689811", "n01498041",
    "n02107312", "n01632458", "n03394916", "n04147183", "n04418357", "n03218198",
    "n01917289", "n02102318", "n02088364", "n09835506", "n02095570", "n03982430",
    "n04041544", "n04562935", "n03933933", "n01843065", "n02128925", "n02480495",
    "n03425413", "n03935335", "n02971356", "n02124075", "n07714571", "n03133878",
    "n02097130", "n02113799", "n09399592", "n03594945",
]

IMAGENET100_CLASSNAMES: List[str] = [
    "palace", "chain saw", "cornet", "maillot", "bell cote", "Cardigan Welsh Corgi",
    "African grey", "sliding door", "white stork", "sunglasses", "dial telephone",
    "flagpole", "cradle", "albatross", "jay", "chiffonier", "birdhouse",
    "swimming trunks", "snail", "paper towel", "race car", "spider monkey",
    "ostrich", "ocarina", "sewing machine", "tusker", "overskirt",
    "desktop computer", "jellyfish", "four-poster bed", "pencil sharpener",
    "red-breasted merganser", "polar bear", "frying pan", "trimaran", "marimba",
    "dowitcher", "wolf spider", "bath towel", "hen of the woods mushroom",
    "water ouzel", "sarong", "electric locomotive", "passenger car", "power drill",
    "crossword puzzle", "Samoyed", "gong", "lighter", "syringe", "pickelhaube",
    "snow leopard", "hare", "ice cream", "Komodo dragon", "guenon", "kelpie (dog)",
    "rifle", "crate", "wardrobe", "shipwreck", "colobus monkey", "fur coat",
    "ladle", "otterhound", "greenhouse", "cockroach", "gibbon", "tow truck",
    "bookshop", "alligator lizard", "stingray", "miniature pinscher",
    "spotted salamander", "French horn", "schooner", "theater curtain",
    "dogsled", "brain coral", "cocker spaniel", "beagle", "baseball player",
    "Lakeland terrier", "pool table", "radio", "water tower", "pier", "jacamar",
    "jaguar", "orangutan", "gas pump", "piggy bank", "carton", "Egyptian cat",
    "head cabbage", "slow cooker", "giant schnauzer", "standard poodle",
    "promontory", "jeep",
]


class ImageNet100:
    """Few-shot ImageNet-100 wrapper compatible with Tip-Adapter.

    The class reuses an existing ImageNet-1k directory (no duplicate copy
    required). Only samples whose directory name (WNID) is in
    ``IMAGENET100_WNIDS`` are kept and relabelled to the range ``[0, 99]``. The
    human-readable names for those classes are provided by
    ``IMAGENET100_CLASSNAMES``.
    """

    dataset_dir = "ILSVRC2012"

    def __init__(
        self,
        root: str,
        num_shots: int,
        preprocess,
        selected_json: str = None,
        strict: bool = True,
        dataset_dir: str = None,
    ) -> None:
        base_dir = dataset_dir or self.dataset_dir
        self.dataset_dir = self._resolve_dataset_dir(root, base_dir)
        self.image_dir = self.dataset_dir

        selected_wnids = self._validate_subset(IMAGENET100_WNIDS)
        self.selected_wnids = selected_wnids

        train_preprocess = preprocess
        test_preprocess = preprocess

        self.train = ImageFolder(
            root=os.path.join(self.image_dir, "train"),
            transform=train_preprocess,
        )
        self.val = ImageFolder(
            root=os.path.join(self.image_dir, "val"),
            transform=test_preprocess,
        )
        self.test = self.val

        self._filter_to_subset(self.train, selected_wnids)
        self._filter_to_subset(self.val, selected_wnids)

        self.template = imagenet_templates
        self.classnames = IMAGENET100_CLASSNAMES

        if selected_json:
            manifest = self._load_manifest(selected_json)
            imgs, targets = self._gather_selected_samples(manifest["per_class"], strict=strict)
            if not imgs:
                raise RuntimeError(
                    f"No valid training samples were collected from {selected_json}. "
                    "Set strict=False to skip missing files if necessary."
                )
            self.manifest = manifest
            self.selected_json = os.path.abspath(os.path.expanduser(selected_json))
            self.train.samples = imgs
            self.train.imgs = imgs
            self.train.targets = targets
        else:
            split_by_label: Dict[int, List[Tuple[str, int]]] = defaultdict(list)
            for path, label in self.train.samples:
                split_by_label[label].append((path, label))

            sampled_imgs: List[Tuple[str, int]] = []
            sampled_targets: List[int] = []

            num_classes = len(selected_wnids)
            for label in range(num_classes):
                items = split_by_label.get(label)
                if items is None:
                    raise RuntimeError(
                        f"Class index {label} is missing after subset filtering."
                    )
                if len(items) < num_shots:
                    raise ValueError(
                        f"Class index {label} only has {len(items)} samples; "
                        f"requires at least {num_shots}."
                    )
                chosen = random.sample(items, num_shots)
                sampled_imgs.extend(chosen)
                sampled_targets.extend([label] * num_shots)

            self.train.samples = sampled_imgs
            self.train.imgs = sampled_imgs
            self.train.targets = sampled_targets

    def _resolve_dataset_dir(self, root: str, dataset_dir: str) -> str:
        candidates = [dataset_dir]
        if dataset_dir == "imagenet100":
            candidates.extend(["ILSVRC2012", "imagenet", "ISLVRC2012"])
        else:
            candidates.extend(["ILSVRC2012", "imagenet"])

        seen = set()
        for candidate in candidates:
            if candidate in seen:
                continue
            seen.add(candidate)
            candidate_path = os.path.join(root, candidate)
            if os.path.isdir(os.path.join(candidate_path, "train")):
                return candidate_path

        return os.path.join(root, dataset_dir)

    def _filter_to_subset(self, dataset: ImageFolder, wnids: List[str]) -> None:
        original_class_to_idx = dataset.class_to_idx
        missing = [wnid for wnid in wnids if wnid not in original_class_to_idx]
        if missing:
            raise ValueError(
                "The following WNIDs are missing from the ImageNet directory: "
                + ", ".join(missing)
            )

        new_class_to_idx = {wnid: idx for idx, wnid in enumerate(wnids)}
        original_idx_to_new = {
            original_class_to_idx[wnid]: new_class_to_idx[wnid] for wnid in wnids
        }

        new_samples = [
            (path, original_idx_to_new[target])
            for path, target in dataset.samples
            if target in original_idx_to_new
        ]

        if len(new_samples) == 0:
            raise RuntimeError("Subset filtering removed all samples.")

        dataset.samples = new_samples
        dataset.imgs = new_samples
        dataset.targets = [target for _, target in new_samples]
        dataset.classes = wnids
        dataset.class_to_idx = new_class_to_idx

    def _load_manifest(self, selected_json: str) -> Dict:
        selected_json = os.path.abspath(os.path.expanduser(selected_json))
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

        for wnid in self.selected_wnids:
            if wnid not in per_class:
                missing_classes.append(wnid)
                continue
            label = self.train.class_to_idx[wnid]
            for path in per_class[wnid]:
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
                "The following classes were not found in the manifest: "
                + ", ".join(sorted(missing_classes))
            )

        if missing_files and not strict:
            print(
                "[ImageNet100] Warning: skipped {} entries missing on disk.".format(
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

    @staticmethod
    def _validate_subset(wnids: Iterable[str]) -> List[str]:
        subset = [str(wnid).strip() for wnid in wnids if str(wnid).strip()]
        duplicates = {wnid for wnid in subset if subset.count(wnid) > 1}
        if duplicates:
            raise ValueError(
                "Duplicate WNIDs found in subset: " + ", ".join(sorted(duplicates))
            )
        if len(subset) != 100:
            raise ValueError(
                f"Subset must contain exactly 100 classes; got {len(subset)}."
            )
        return subset
