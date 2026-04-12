import json
import os
import pickle
from collections import OrderedDict, defaultdict

from dassl.data.datasets import DATASET_REGISTRY, Datum, DatasetBase
from dassl.utils import listdir_nohidden, mkdir_if_missing

from .oxford_pets import OxfordPets

IMAGENET100_WNIDS = [
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

BUILTIN_IMAGENET_SUBSETS = {
    "imagenet100": IMAGENET100_WNIDS,
}


@DATASET_REGISTRY.register()
class ImageNet(DatasetBase):

    dataset_dir = "imagenet"

    def __init__(self, cfg):
        root = os.path.abspath(os.path.expanduser(cfg.DATASET.ROOT))
        dataset_subdir = cfg.DATASET.IMAGENET_DIR.strip() or self.dataset_dir
        self.dataset_dir = os.path.join(root, dataset_subdir)
        images_subdir = cfg.DATASET.IMAGENET_IMAGES_SUBDIR.strip()
        if images_subdir:
            self.image_dir = os.path.join(self.dataset_dir, images_subdir)
        else:
            self.image_dir = self.dataset_dir
        default_classnames = os.path.join(self.dataset_dir, "classnames.txt")
        classnames_file = cfg.DATASET.CLASSNAMES_TXT.strip()
        if classnames_file:
            classnames_source = os.path.abspath(os.path.expanduser(classnames_file))
        else:
            classnames_source = default_classnames

        builtin_subset = cfg.DATASET.IMAGENET_BUILTIN_SUBSET.strip().lower()
        builtin_wnids = None
        subset_tag = "default"
        if builtin_subset:
            if builtin_subset not in BUILTIN_IMAGENET_SUBSETS:
                raise ValueError(
                    f"Unknown built-in ImageNet subset '{builtin_subset}'. "
                    f"Available: {list(BUILTIN_IMAGENET_SUBSETS.keys())}"
                )
            builtin_wnids = BUILTIN_IMAGENET_SUBSETS[builtin_subset]
            subset_tag = builtin_subset
        else:
            subset_tag = self._determine_subset_tag(classnames_source, default_classnames)

        cache_suffix = "" if subset_tag == "default" else f"_{subset_tag}"
        self.preprocessed = os.path.join(self.dataset_dir, f"preprocessed{cache_suffix}.pkl")
        self.split_fewshot_dir = os.path.join(self.dataset_dir, f"split_fewshot{cache_suffix}")
        mkdir_if_missing(self.split_fewshot_dir)
        if cache_suffix:
            print(f"Using ImageNet subset cache namespace '{subset_tag}' "
                  f"(preprocessed -> {os.path.basename(self.preprocessed)})")

        if os.path.exists(self.preprocessed):
            with open(self.preprocessed, "rb") as f:
                preprocessed = pickle.load(f)
                train = preprocessed["train"]
                test = preprocessed["test"]
        else:
            if builtin_wnids:
                classnames = self._build_classnames_from_builtin(builtin_wnids, classnames_source)
            else:
                if not os.path.exists(classnames_source):
                    raise FileNotFoundError(
                        f"Classnames file not found at {classnames_source}. "
                        "Set cfg.DATASET.CLASSNAMES_TXT to the correct path."
                    )
                classnames = self.read_classnames(classnames_source)
            train = self.read_data(classnames, "train")
            # Follow standard practice to perform evaluation on the val set
            # Also used as the val set (so evaluate the last-step model)
            test = self.read_data(classnames, "val")

            preprocessed = {"train": train, "test": test}
            with open(self.preprocessed, "wb") as f:
                pickle.dump(preprocessed, f, protocol=pickle.HIGHEST_PROTOCOL)

        num_shots = cfg.DATASET.NUM_SHOTS
        if num_shots >= 1:
            seed = cfg.SEED
            preprocessed = os.path.join(self.split_fewshot_dir, f"shot_{num_shots}-seed_{seed}.pkl")
            
            if os.path.exists(preprocessed):
                print(f"Loading preprocessed few-shot data from {preprocessed}")
                with open(preprocessed, "rb") as file:
                    data = pickle.load(file)
                    train = data["train"]
            else:
                train = self.generate_fewshot_dataset(train, num_shots=num_shots)
                data = {"train": train}
                print(f"Saving preprocessed few-shot data to {preprocessed}")
                with open(preprocessed, "wb") as file:
                    pickle.dump(data, file, protocol=pickle.HIGHEST_PROTOCOL)
            self.export_fewshot_filenames(train, cfg, preprocessed)

        subsample = cfg.DATASET.SUBSAMPLE_CLASSES
        train, test = OxfordPets.subsample_classes(train, test, subsample=subsample)

        super().__init__(train_x=train, val=test, test=test)

    @staticmethod
    def _build_classnames_from_builtin(wnids, fallback_file):
        """Construct classnames dict using built-in WNIDs and optional text file."""
        base = {}
        if fallback_file and os.path.exists(fallback_file):
            base = ImageNet.read_classnames(fallback_file)

        classnames = OrderedDict()
        missing = []
        for wnid in wnids:
            if wnid in base:
                classnames[wnid] = base[wnid]
            else:
                classnames[wnid] = wnid
                missing.append(wnid)

        if missing and fallback_file:
            print(
                f"Warning: {len(missing)} WNIDs from the built-in subset are not present in "
                f"{fallback_file}. Using raw IDs as class names (e.g., {missing[0]})."
            )
        return classnames

    @staticmethod
    def read_classnames(text_file):
        """Return a dictionary containing
        key-value pairs of <folder name>: <class name>.
        """
        classnames = OrderedDict()
        with open(text_file, "r") as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip().split(" ")
                folder = line[0]
                classname = " ".join(line[1:])
                classnames[folder] = classname
        return classnames

    def read_data(self, classnames, split_dir):
        split_dir = os.path.join(self.image_dir, split_dir)
        available = {f.name for f in os.scandir(split_dir) if f.is_dir()}

        requested = list(classnames.keys())
        selected = [folder for folder in requested if folder in available]
        if not selected:
            raise RuntimeError(
                f"No class folders from {len(requested)} requested classes were found in {split_dir}. "
                "Check cfg.DATASET.CLASSNAMES_TXT or the dataset layout."
            )

        missing = [folder for folder in requested if folder not in available]
        if missing:
            print(f"Warning: skipping {len(missing)} classes that are listed but not found in {split_dir}. "
                  f"Example missing class: {missing[0]}")

        extras = sorted(available - set(requested))
        if extras:
            print(f"Info: found {len(extras)} extra class folders not listed in cfg.DATASET.CLASSNAMES_TXT. "
                  "They will be ignored.")

        items = []
        for label, folder in enumerate(selected):
            imnames = listdir_nohidden(os.path.join(split_dir, folder))
            classname = classnames[folder]
            for imname in imnames:
                impath = os.path.join(split_dir, folder, imname)
                item = Datum(impath=impath, label=label, classname=classname)
                items.append(item)

        return items

    @staticmethod
    def export_fewshot_filenames(train, cfg, cache_file):
        """Persist shot file names per class for reproducibility."""
        output_dir = getattr(cfg, "OUTPUT_DIR", "")
        if not output_dir:
            return

        mkdir_if_missing(output_dir)
        per_class = defaultdict(list)
        for item in train:
            folder = os.path.basename(os.path.dirname(item.impath))
            per_class[folder].append(item.impath)
        per_class = dict(per_class)

        save_path = os.path.join(output_dir, "selected_shots.json")
        payload = {
            "seed": cfg.SEED,
            "num_shots": cfg.DATASET.NUM_SHOTS,
            "cache_file": cache_file,
            "total_examples": len(train),
            "per_class": per_class,
        }
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        print(f"Saved few-shot filename manifest to {save_path}")

    @staticmethod
    def _determine_subset_tag(classnames_path, default_path):
        """Return a stable identifier for the current class subset."""
        norm = os.path.abspath(os.path.expanduser(classnames_path))
        default_norm = os.path.abspath(os.path.expanduser(default_path))
        if norm == default_norm:
            return "default"

        base = os.path.splitext(os.path.basename(classnames_path))[0]
        sanitized = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in base)
        return sanitized or "custom"
