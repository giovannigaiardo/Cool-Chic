import subprocess
from argparse import ArgumentParser
from pathlib import Path

BASE_COMMAND = ["python3", "./cc_encode.py", "--debug"]


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
    for img in input_folder.rglob("*.png"):
        atomic_arguments = [
            "--input",
            img.as_posix(),
            "--workdir",
            output_folder.as_posix(),
            "--output",
            img.stem + ".cool",
        ]
        commands.append(BASE_COMMAND + atomic_arguments)
    return commands


def run_commands(commands: list):
    for cmd in commands:
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            print(f"Success: {' '.join(cmd)}")
            print(result.stdout)
        except subprocess.CalledProcessError as e:
            print(f"Could not run {' '.join(cmd)}:\n {e}")


def main():
    args = create_parser().parse_args()
    commands = build_commands(args.input_folder, args.output_folder)
    run_commands(commands)


if __name__ == "__main__":
    main()
