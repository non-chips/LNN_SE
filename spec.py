import argparse
import os
from pathlib import Path

import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
from omegaconf import OmegaConf


TIERS = {
    "low": 0.10,
    "mid": 0.50,
    "high": 0.90,
}


def read_score_scp(path):
    scores = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            uid, score = line.strip().split()
            scores[uid] = float(score)
    return scores


def select_examples(pesq_scores):
    items = sorted(pesq_scores.items(), key=lambda x: x[1])
    values = np.array([score for _, score in items])

    selected = {}
    used = set()
    for tier, q in TIERS.items():
        target = np.quantile(values, q)
        for uid, _ in sorted(items, key=lambda x: abs(x[1] - target)):
            if uid not in used:
                selected[tier] = uid
                used.add(uid)
                break
    return selected


def load_audio(path):
    audio, sr = sf.read(path, dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    return audio, sr


def wav_to_mag(audio, nfft, hop):
    return np.abs(librosa.stft(audio, n_fft=nfft, hop_length=hop, win_length=nfft, window="hann"))


def plot_spec(ax, spec_db, sr, hop, title, metrics=None, cmap="magma"):
    img = librosa.display.specshow(spec_db, x_axis="time", y_axis="linear", sr=sr, hop_length=hop, cmap=cmap, ax=ax)
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (Hz)")

    if metrics is not None:
        ax.text(
            0.98,
            0.96,
            metrics,
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=9,
            color="white",
            bbox={"facecolor": "black", "alpha": 0.6, "edgecolor": "none", "pad": 4},
        )
    return img


def main(args):
    cfg = OmegaConf.load(args.config)
    clean_dir = Path(cfg.test_dataset.clean_dir)
    noisy_dir = Path(cfg.test_dataset.noisy_dir)
    enh_dir = Path(cfg.network.enh_folder)
    out_dir = enh_dir / args.output_dir
    os.makedirs(out_dir, exist_ok=True)

    enh_pesq = read_score_scp(enh_dir / "scoring_intrusive" / "PESQ.scp")
    enh_estoi = read_score_scp(enh_dir / "scoring_intrusive" / "ESTOI.scp")
    noisy_pesq = read_score_scp(enh_dir / "scoring_intrusive_noisy" / "PESQ.scp")
    noisy_estoi = read_score_scp(enh_dir / "scoring_intrusive_noisy" / "ESTOI.scp")
    selected = select_examples(enh_pesq)

    fig, axes = plt.subplots(3, 3, figsize=(12, 10), constrained_layout=True, sharey=True)
    for i, name in enumerate(["Clean", "Noisy", "Enhanced"]):
        axes[0, i].annotate(name, xy=(0.5, 1.20), xycoords="axes fraction", ha="center", fontsize=13, fontweight="bold")

    selected_lines = []
    last_img = None
    for row, (tier, uid) in enumerate(selected.items()):
        clean, sr = load_audio(clean_dir / f"{uid}.wav")
        noisy, _ = load_audio(noisy_dir / f"{uid}.wav")
        enhanced, _ = load_audio(enh_dir / f"{uid}_enh.wav")

        mags = [wav_to_mag(x, args.nfft, args.hop) for x in [clean, noisy, enhanced]]
        ref = max(float(x.max()) for x in mags)
        specs = [librosa.amplitude_to_db(x, ref=ref, top_db=args.top_db) for x in mags]

        noisy_metrics = f"PESQ={noisy_pesq[uid]:.3f}\nESTOI={noisy_estoi[uid]:.3f}"
        enh_metrics = f"PESQ={enh_pesq[uid]:.3f}\nESTOI={enh_estoi[uid]:.3f}"

        last_img = plot_spec(axes[row, 0], specs[0], sr, args.hop, f"{tier.upper()} | {uid} clean", cmap=args.cmap)
        plot_spec(axes[row, 1], specs[1], sr, args.hop, f"{tier.upper()} | {uid} noisy", noisy_metrics, args.cmap)
        plot_spec(axes[row, 2], specs[2], sr, args.hop, f"{tier.upper()} | {uid} enhanced", enh_metrics, args.cmap)

        selected_lines.append(
            f"{tier} {uid} "
            f"noisy_PESQ={noisy_pesq[uid]:.6f} noisy_ESTOI={noisy_estoi[uid]:.6f} "
            f"enhanced_PESQ={enh_pesq[uid]:.6f} enhanced_ESTOI={enh_estoi[uid]:.6f}"
        )

    fig.colorbar(last_img, ax=axes, location="right", shrink=0.88, pad=0.01, label="Magnitude (dB)")

    fig_path = out_dir / "pesq_low_mid_high_spectrograms.png"
    fig.savefig(fig_path, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)

    selected_path = out_dir / "selected_examples.txt"
    with open(selected_path, "w", encoding="utf-8", newline="\n") as f:
        for line in selected_lines:
            f.write(line + "\n")

    print(f"Figure saved to: {fig_path.as_posix()}")
    print(f"Selected examples saved to: {selected_path.as_posix()}")
    for line in selected_lines:
        print(line)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-C", "--config", default="configs/cfg_infer.yaml")
    parser.add_argument("--output-dir", default="spectrogram_examples")
    parser.add_argument("--nfft", type=int, default=512)
    parser.add_argument("--hop", type=int, default=256)
    parser.add_argument("--top-db", type=float, default=100.0)
    parser.add_argument("--cmap", default="magma")
    parser.add_argument("--dpi", type=int, default=220)

    args = parser.parse_args()
    main(args)