import argparse
import logging
import os
from datetime import datetime
from pathlib import Path

LOGGER_NAME = "gp_adapter_imagenet"
REPO_ROOT = Path(__file__).resolve().parent
CONFIG_ROOT = REPO_ROOT / "datasets" / "configs"
DEFAULT_MANIFEST_ROOT = REPO_ROOT / "manifests"


def configure_logger(logger, log_file=None, log_dir="logs", config_path=None, run_tag=None):
    """Configure logger with console and file handlers."""
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    log_dir = Path(log_dir)

    if log_file:
        log_path = Path(log_file)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        cfg_name = Path(config_path).stem if config_path else "run"
        sanitized_tag = f"_{run_tag}" if run_tag else ""
        log_path = log_dir / f"{cfg_name}{sanitized_tag}_{timestamp}.log"

    log_path.parent.mkdir(parents=True, exist_ok=True)

    file_fmt = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    console_fmt = logging.Formatter("%(message)s")

    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(file_fmt)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(console_fmt)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    logger.propagate = False
    return log_path


def create_print_logger(active_logger):
    """Override print to funnel into the logger for unified logging."""
    def log_print(*values, **kwargs):
        sep = kwargs.get("sep", " ")
        end = kwargs.get("end", "\n")
        message = sep.join(str(v) for v in values) if values else ""
        if end and end != "\n":
            message = f"{message}{end}"
        active_logger.info(message)
    return log_print


def parse_loss_thr(value):
    value_str = str(value).strip().lower()
    if value_str in {"inf", "+inf", "infinity", "+infinity"}:
        return float("inf")
    if value_str in {"-inf", "-infinity"}:
        return float("-inf")
    return float(value)


def build_arg_parser():
    parser = argparse.ArgumentParser(description="Run GP experiments on ImageNet data.")
    parser.add_argument('--log-dir', default='./logs', help='Directory to store log files (default: ./logs)')
    parser.add_argument('--log-file', default=None, help='Optional explicit log file path')
    parser.add_argument('--config', default=None, help='Path to config file (default: imagenet or imagenet100 config)')
    parser.add_argument('--root-path', default=os.environ.get("DATA_ROOT"), help='Dataset root path overriding root_path in the config')
    parser.add_argument('--root-path-ood', default=os.environ.get("OOD_DATA_ROOT"), help='OOD dataset root path overriding root_path_ood in the config')
    parser.add_argument('--data-dir-name', default=None, help='Dataset directory name (default varies by dataset)')
    parser.add_argument('--imagenet100', action='store_true', help='Use ImageNet-100 settings')
    parser.add_argument('--seed', type=int, default=int(os.environ.get("TIP_SEED", 2)), help='Random seed for shot selection and reproducibility')
    parser.add_argument('--trainer', choices=['LoCoOp', 'CoOp', 'NoPrompt'], default=os.environ.get("TIP_TRAINER", "LoCoOp"), help='Prompting strategy to associate with the run')
    parser.add_argument('--backbone', default=os.environ.get("TIP_BACKBONE"), help='Override backbone defined in the config (e.g., RN50 or ViT-B/16)')
    parser.add_argument('--shots', type=int, default=None, help='Override the number of shots defined in the config')
    parser.add_argument('--mix-alpha', type=float, default=float(os.environ.get("TIP_MIX_ALPHA", 0.15)), help='Mixing ratio between image/text GP outputs (default: 0.15)')
    parser.add_argument('--loss-thr', type=parse_loss_thr, default=float(os.environ.get("TIP_LOSS_THR", -5)), help='Log-likelihood threshold used in kernel optimization (default: -5)')
    parser.add_argument('--locoop-output-root', default=os.environ.get("LOCOOP_OUTPUT_ROOT"), help='Root directory containing LoCoOp outputs/checkpoints')
    parser.add_argument('--locoop-config-dir', default=os.environ.get("LOCOOP_CONFIG_DIR"), help='Directory containing LoCoOp yaml configs such as vit_b16_ep50.yaml')
    parser.add_argument('--selected-shots-json', default=os.environ.get("SELECTED_SHOTS_JSON"), help='Optional explicit path to selected_shots.json')
    parser.add_argument('--manifest-root', default=os.environ.get("MANIFEST_ROOT"), help='Root directory containing trainer-independent selected_shots manifests')
    return parser
