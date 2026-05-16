import os
import torch
import soundfile as sf
from tqdm import tqdm
from omegaconf import OmegaConf
from models.LNN_end2end import LNN as Model
from evaluation.rtf_evaluate import RTFTracker


def main(args):
    cfg_infer = OmegaConf.load(args.config)
    cfg_network = OmegaConf.load(cfg_infer.network.config)

    noisy_folder = cfg_infer.test_dataset.noisy_dir
    clean_folder = cfg_infer.test_dataset.clean_dir
    enh_folder = cfg_infer.network.enh_folder
    os.makedirs(enh_folder, exist_ok=True)

    device = torch.device(f"cuda:{args.device}" if args.device != "cpu" and torch.cuda.is_available() else "cpu")

    model = Model(**cfg_network["network_config"]).to(device)
    checkpoint = torch.load(cfg_infer.network.checkpoint, map_location=device)
    state_dict = checkpoint["model"] if "model" in checkpoint else checkpoint
    state_dict = {key.removeprefix("module."): value for key, value in state_dict.items()}
    model.load_state_dict(state_dict)
    model.eval()

    if "scp" in cfg_infer.test_dataset and cfg_infer.test_dataset.scp:
        with open(cfg_infer.test_dataset.scp, "r", encoding="utf-8") as f:
            noisy_wavs = [line.strip().split()[0] + ".wav" for line in f if line.strip()]
    else:
        noisy_wavs = sorted(list(filter(lambda x: x.endswith("wav"), os.listdir(noisy_folder))))

    rtf_tracker = RTFTracker(args.save_rtf, enh_folder, device)
    inf_scp_list = []
    ref_scp_list = []
    for wav_name in tqdm(noisy_wavs):
        noisy_path = os.path.join(noisy_folder, wav_name)
        clean_path = os.path.join(clean_folder, wav_name)
        noisy, fs = sf.read(noisy_path, dtype="float32")
        if noisy.ndim > 1:
            noisy = noisy.mean(axis=1)

        input = torch.FloatTensor(noisy).unsqueeze(0).to(device)
        start_time = rtf_tracker.start()
        with torch.inference_mode():
            output = model(input)
        processing_time = rtf_tracker.stop(start_time)
        enhanced = output.cpu().detach().numpy().squeeze()

        uid = wav_name.split(".wav")[0]
        enh_path = os.path.join(enh_folder, uid + "_enh.wav")

        inf_scp_list.append([uid, enh_path])
        ref_scp_list.append([uid, clean_path])

        sf.write(enh_path, enhanced, fs)
        rtf_tracker.add(uid, len(noisy), fs, processing_time)

    with open(os.path.join(enh_folder, "inf.scp"), "w", encoding="utf-8", newline="\n") as f:
        for uid, audio_path in inf_scp_list:
            f.write(f"{uid} {audio_path}\n")

    with open(os.path.join(enh_folder, "ref.scp"), "w", encoding="utf-8", newline="\n") as f:
        for uid, audio_path in ref_scp_list:
            f.write(f"{uid} {audio_path}\n")

    rtf_tracker.save()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-C", "--config", default="configs/cfg_infer.yaml")
    parser.add_argument("-D", "--device", default="0", help="Index of the gpu device, or cpu")
    parser.add_argument("--save-rtf", action="store_true", help="Save per-utterance RTF and summary during inference")

    args = parser.parse_args()
    main(args)