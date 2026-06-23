from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path
import shutil


def _extract_step(path: Path) -> int:
    try:
        return int(path.name.split("global_step_", 1)[1])
    except Exception as e:  # pragma: no cover
        raise ValueError(f"Invalid checkpoint directory name: {path}") from e


def _find_latest_checkpoint(checkpoint_root: Path) -> Path:
    candidates = [path for path in checkpoint_root.iterdir() if path.is_dir() and path.name.startswith("global_step_")]
    if not candidates:
        raise FileNotFoundError(f"No global_step_* checkpoints found under {checkpoint_root}")
    return max(candidates, key=_extract_step)


def main() -> None:
    parser = ArgumentParser(description="Promote the latest SFT global_step checkpoint to a stable merged model directory.")
    parser.add_argument("--checkpoint_root", type=Path, required=True, help="Directory containing global_step_* subdirectories.")
    parser.add_argument("--output_dir", type=Path, required=True, help="Stable model directory to materialize.")
    parser.add_argument("--symlink", action="store_true", help="Create a symlink instead of copying files.")
    parser.add_argument("--keep_existing", action="store_true", help="Refuse to replace output_dir if it already exists.")
    args = parser.parse_args()

    if not args.checkpoint_root.exists():
        raise SystemExit(f"Checkpoint root not found: {args.checkpoint_root}")

    latest = _find_latest_checkpoint(args.checkpoint_root)

    if args.output_dir.exists() or args.output_dir.is_symlink():
        if args.keep_existing:
            raise SystemExit(f"Output dir already exists: {args.output_dir}")
        if args.output_dir.is_symlink() or args.output_dir.is_file():
            args.output_dir.unlink()
        else:
            shutil.rmtree(args.output_dir)

    args.output_dir.parent.mkdir(parents=True, exist_ok=True)

    if args.symlink:
        args.output_dir.symlink_to(latest.resolve(), target_is_directory=True)
    else:
        shutil.copytree(latest, args.output_dir)

    print(f"Latest checkpoint: {latest}")
    print(f"Merged model dir: {args.output_dir}")


if __name__ == "__main__":
    main()
