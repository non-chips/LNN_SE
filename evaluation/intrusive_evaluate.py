import logging
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
from pesq import PesqError, pesq
from pystoi import stoi
from tqdm import tqdm

try:
    from p_tqdm import p_map
except ImportError:
    p_map = None


METRICS = ("SNR", "SISNR", "PESQ", "ESTOI")


def estoi_metric(ref, inf, fs=16000):
    return stoi(ref, inf, fs_sig=fs, extended=True)


def pesq_metric(ref, inf, fs=8000):
    if fs == 8000:
        mode = "nb"
    elif fs == 16000:
        mode = "wb"
    elif fs > 16000:
        mode = "wb"
        ref = librosa.resample(ref, orig_sr=fs, target_sr=16000)
        inf = librosa.resample(inf, orig_sr=fs, target_sr=16000)
        fs = 16000
    else:
        raise ValueError(f"sample rate must be 8000 or 16000+ for PESQ evaluation, but got {fs}")

    pesq_score = pesq(fs, ref, inf, mode=mode, on_error=PesqError.RETURN_VALUES)
    if pesq_score == PesqError.NO_UTTERANCES_DETECTED:
        logging.warning("[PESQ] Error: No utterances detected. Skipping this sample.")
        return None
    return pesq_score


def sisnr_metric(ref, inf):
    inf = inf - inf.mean()
    ref = ref - ref.mean()
    scale = np.sum(inf * ref) / (np.sum(ref**2) + 1e-8)
    target = scale * ref
    residual = inf - target
    return 10 * np.log10((np.sum(target**2) + 1e-8) / (np.sum(residual**2) + 1e-8))


def snr_metric(ref, inf):
    inf = inf - inf.mean()
    ref = ref - ref.mean()
    residual = inf - ref
    return 10 * np.log10((np.sum(ref**2) + 1e-8) / (np.sum(residual**2) + 1e-8))


def read_scp(scp_path):
    entries = {}
    with open(scp_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split(maxsplit=1)
            if not parts:
                continue
            if len(parts) != 2:
                raise ValueError(f"Invalid scp line in {scp_path}: {line.rstrip()}")
            uid, audio_path = parts
            entries[uid] = audio_path
    return entries


def main(args):
    refs = read_scp(args.ref_scp)
    infs = read_scp(args.inf_scp)
    common_uids = sorted(set(refs) & set(infs))
    if not common_uids:
        raise ValueError("No matching utterance ids between ref_scp and inf_scp.")

    data_pairs = [(uid, refs[uid], infs[uid]) for uid in common_uids]
    if args.nj <= 1 or p_map is None:
        ret = [process_one_pair(data_pair) for data_pair in tqdm(data_pairs)]
    else:
        ret = p_map(process_one_pair, data_pairs, num_cpus=args.nj)

    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    writers = {metric: (outdir / f"{metric}.scp").open("w", encoding="utf-8", newline="\n") for metric in METRICS}

    for uid, score in ret:
        for metric, value in score.items():
            writers[metric].write(f"{uid} {value}\n")

    for writer in writers.values():
        writer.close()

    with (outdir / "RESULTS.txt").open("w", encoding="utf-8", newline="\n") as f:
        for metric in METRICS:
            mean_score = np.nanmean([score[metric] for _, score in ret])
            f.write(f"{metric}: {mean_score:.4f}\n")

    print(f"Overall results have been written in {outdir / 'RESULTS.txt'}", flush=True)


def process_one_pair(data_pair):
    uid, ref_path, inf_path = data_pair
    ref, fs = sf.read(ref_path, dtype="float32")
    inf, fs2 = sf.read(inf_path, dtype="float32")
    if fs != fs2:
        raise ValueError(f"Sample rate mismatch for {uid}: ref={fs}, inf={fs2}")

    if ref.ndim != inf.ndim:
        raise ValueError(
            f"Audio dimension mismatch for {uid}: ref_shape={ref.shape}, inf_shape={inf.shape}, "
            f"ref_path={ref_path}, inf_path={inf_path}"
        )
    if ref.shape != inf.shape:
        raise ValueError(
            f"Audio length/shape mismatch for {uid}: ref_shape={ref.shape}, inf_shape={inf.shape}, "
            f"ref_path={ref_path}, inf_path={inf_path}"
        )
    if ref.size == 0:
        raise ValueError(f"Empty audio for {uid}: ref_path={ref_path}, inf_path={inf_path}")

    if ref.ndim > 1:
        ref = ref.mean(axis=1)
        inf = inf.mean(axis=1)

    scores = {}
    for metric in METRICS:
        if metric == "PESQ":
            pesq_score = pesq_metric(ref, inf, fs=fs)
            scores[metric] = pesq_score if pesq_score is not None else np.nan
        elif metric == "ESTOI":
            scores[metric] = estoi_metric(ref, inf, fs=fs)
        elif metric == "SISNR":
            scores[metric] = sisnr_metric(ref, inf)
        elif metric == "SNR":
            scores[metric] = snr_metric(ref, inf)
        else:
            raise NotImplementedError(metric)

    return uid, scores


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--ref_scp", type=str, required=True, help="Path to the scp file containing reference signals")
    parser.add_argument("--inf_scp", type=str, required=True, help="Path to the scp file containing enhanced signals")
    parser.add_argument("--output_dir", type=str, required=True, help="Path to the output directory for writing metrics")
    parser.add_argument("--nj", type=int, default=4, help="Number of parallel workers to speed up evaluation")
    parser.add_argument("--chunksize", type=int, default=1000, help="Kept for compatibility with evaluate.py")
    args = parser.parse_args()
    main(args)
