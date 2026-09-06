import logging
import subprocess
import typing
from argparse import ArgumentParser
from datetime import UTC, datetime
from itertools import product
from pathlib import Path

import numpy as np
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

from coolchic.training.loss import DISTORTION_METRIC

script_dir = Path(__file__).resolve().parent
log_file = script_dir / f"./{datetime.now(tz=UTC).timestamp()}.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.FileHandler(log_file)],
)
logger = logging.getLogger(__name__)

BASE_COMMAND = ["python3", "./cc_encode.py"]
N_ITR = int(1e4)
losses = typing.get_args(DISTORTION_METRIC)
lmbdas = np.logspace(-5, -1, num=7)


def create_parser() -> ArgumentParser:
    parser = ArgumentParser(
        description="Script to benchmark different versions of Cool-Chic."
    )
    parser.add_argument(
        "-i",
        "--input_folder",
        help="Path to the folder containing your test images",
        type=Path,
    )
    parser.add_argument(
        "-o",
        "--output_folder",
        help="Path to the folder where you wish to store experimental results",
        type=Path,
        default="./test",
    )
    return parser


def build_commands(input_folder: Path, output_folder: Path) -> list:
    commands = []
    for loss, lmbda, img in product(losses, lmbdas, input_folder.rglob("*.png")):
        experiment_name = f"{loss}_{lmbda:.1e}_{img.stem}"
        workdir = output_folder / experiment_name
        workdir.mkdir(parents=True, exist_ok=True)
        atomic_arguments = [
            "--input",
            img.as_posix(),
            "--workdir",
            workdir.as_posix(),
            "--output",
            experiment_name + ".cool",
            "--tune",
            loss,
            "--lmbda",
            str(lmbda),
            "--n_itr",
            str(N_ITR),
        ]
        commands.append(BASE_COMMAND + atomic_arguments)
    return commands


def run_commands(commands: list):
    with logging_redirect_tqdm():
        for cmd in tqdm(commands, desc="Running experiments...", unit="exp"):
            try:
                result = subprocess.run(cmd, capture_output=True, text=True, check=True)
                logger.info(f"Success: {' '.join(cmd)}")

                if result.stdout.strip():
                    logger.info(f"Output: \n{result.stdout.strip()}")
            except subprocess.CalledProcessError as e:
                logger.error(
                    f"Could not run {' '.join(cmd)}\n"
                    f"Return code: {e.returncode}\n"
                    f"Error output: {e.stderr.strip() if e.stderr else e.output}"
                )


def main():
    args = create_parser().parse_args()
    commands = build_commands(args.input_folder, args.output_folder)
    run_commands(commands)


if __name__ == "__main__":
    main()
