from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch


class RTFTracker:
    def __init__(self, enabled: bool, output_dir: str | Path, device: torch.device):
        self.enabled = enabled
        self.output_dir = Path(output_dir)
        self.device = device
        self.rows: list[tuple[str, float, float, float]] = []
        self.total_audio_duration = 0.0
        self.total_processing_time = 0.0

    def _synchronize(self) -> None:
        if self.enabled and self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    def start(self) -> float | None:
        if not self.enabled:
            return None
        self._synchronize()
        return time.perf_counter()

    def stop(self, start_time: float | None) -> float | None:
        if not self.enabled or start_time is None:
            return None
        self._synchronize()
        return time.perf_counter() - start_time

    def add(self, uid: str, num_samples: int, sample_rate: int, processing_time: float | None) -> None:
        if not self.enabled or processing_time is None:
            return
        audio_duration = num_samples / float(sample_rate)
        rtf = processing_time / audio_duration
        self.total_audio_duration += audio_duration
        self.total_processing_time += processing_time
        self.rows.append((uid, audio_duration, processing_time, rtf))

    def save(self) -> None:
        if not self.enabled or not self.rows:
            return

        rtf_csv_path = self.output_dir / "rtf.csv"
        rtf_summary_path = self.output_dir / "rtf_summary.txt"
        rtfs = np.array([row[3] for row in self.rows], dtype=np.float64)
        processing_times_ms = np.array([row[2] * 1000.0 for row in self.rows], dtype=np.float64)
        overall_rtf = self.total_processing_time / self.total_audio_duration

        with rtf_csv_path.open("w", encoding="utf-8", newline="\n") as f:
            f.write("uid,audio_duration_sec,processing_time_sec,processing_time_ms,rtf\n")
            for uid, audio_duration, processing_time, rtf in self.rows:
                f.write(f"{uid},{audio_duration:.6f},{processing_time:.6f},{processing_time * 1000.0:.6f},{rtf:.6f}\n")

        summary_lines = [
            f"num_files: {len(self.rows)}",
            f"total_audio_duration_sec: {self.total_audio_duration:.6f}",
            f"total_processing_time_sec: {self.total_processing_time:.6f}",
            f"overall_rtf: {overall_rtf:.6f}",
            f"mean_rtf: {float(np.mean(rtfs)):.6f}",
            f"max_rtf: {float(np.max(rtfs)):.6f}",
            f"min_rtf: {float(np.min(rtfs)):.6f}",
            f"mean_processing_time_ms: {float(np.mean(processing_times_ms)):.6f}",
            f"max_processing_time_ms: {float(np.max(processing_times_ms)):.6f}",
        ]

        with rtf_summary_path.open("w", encoding="utf-8", newline="\n") as f:
            f.write("\n".join(summary_lines))
            f.write("\n")

        print(f"RTF csv: {rtf_csv_path.as_posix()}")
        print(f"RTF summary: {rtf_summary_path.as_posix()}")