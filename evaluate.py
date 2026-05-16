import os
import subprocess
import sys
from pathlib import Path

from omegaconf import OmegaConf


def make_noisy_baseline_scps(config, enh_folder):
    noisy_dir = Path(config.test_dataset.noisy_dir)
    clean_dir = Path(config.test_dataset.clean_dir)
    scp_path = Path(config.test_dataset.scp) if config.test_dataset.get("scp", None) else None
    if not noisy_dir.is_dir():
        raise NotADirectoryError(f"Noisy test directory not found: {noisy_dir}")
    if not clean_dir.is_dir():
        raise NotADirectoryError(f"Clean test directory not found: {clean_dir}")

    if scp_path is not None:
        if not scp_path.is_file():
            raise FileNotFoundError(f"Test scp not found: {scp_path}")
        with scp_path.open("r", encoding="utf-8") as f:
            uids = [line.strip().split()[0] for line in f if line.strip()]
    else:
        uids = sorted(path.stem for path in noisy_dir.glob("*.wav"))

    baseline_dir = Path(enh_folder) / "noisy_baseline"
    baseline_dir.mkdir(parents=True, exist_ok=True)
    ref_scp = baseline_dir / "ref.scp"
    inf_scp = baseline_dir / "inf.scp"

    with ref_scp.open("w", encoding="utf-8", newline="\n") as ref_f, inf_scp.open("w", encoding="utf-8", newline="\n") as inf_f:
        for uid in uids:
            clean_path = clean_dir / f"{uid}.wav"
            noisy_path = noisy_dir / f"{uid}.wav"
            if not clean_path.is_file():
                raise FileNotFoundError(f"Clean audio not found: {clean_path}")
            if not noisy_path.is_file():
                raise FileNotFoundError(f"Noisy audio not found: {noisy_path}")
            ref_f.write(f"{uid} {clean_path.as_posix()}\n")
            inf_f.write(f"{uid} {noisy_path.as_posix()}\n")

    return ref_scp, inf_scp


def run_intrusive(ref_scp, inf_scp, output_dir):
    subprocess.run(
        [
            sys.executable,
            "./evaluation/intrusive_evaluate.py",
            "--ref_scp",
            str(ref_scp),
            "--inf_scp",
            str(inf_scp),
            "--output_dir",
            str(output_dir),
        ],
        check=True,
    )


def main(args):
    config = OmegaConf.load(args.config)
    enh_folder = Path(config.network.enh_folder)
    if not enh_folder.is_dir():
        raise NotADirectoryError(f"Enhanced audio folder not found: {enh_folder}")

    ref_scp = enh_folder / "ref.scp"
    inf_scp = enh_folder / "inf.scp"

    if args.metric == "intrusive":
        run_intrusive(ref_scp, inf_scp, enh_folder / "scoring_intrusive")
        if args.eval_noisy:
            noisy_ref_scp, noisy_inf_scp = make_noisy_baseline_scps(config, enh_folder)
            run_intrusive(noisy_ref_scp, noisy_inf_scp, enh_folder / "scoring_intrusive_noisy")
    else:
        raise ValueError(f"Unsupported metric: {args.metric}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--metric", required=True, choices=["intrusive"], help="Metric to be calculated")
    parser.add_argument("--config", default="configs/cfg_infer.yaml")
    parser.add_argument("--device", default="0")
    parser.add_argument("--eval-noisy", action="store_true", help="Also evaluate unprocessed noisy speech")

    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device
    main(args)