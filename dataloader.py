from random import random
import soundfile as sf
import librosa
import torch
from torch.utils import data
import numpy as np
import random

NOISY_DATABASE_TRAIN = 'Datasets/noisy_trainset_wav'
NOISY_DATABASE_VALID = 'Datasets/noisy_validation_wav'


class VoiceBank_Demand(torch.utils.data.Dataset):
    def __init__(
        self,
        fs=16000,
        length_in_seconds=2.5,
        num_data_tot=10000,
        num_data_per_epoch=500,
        random_start_point=True,
        train=True
    ):
        if train:
            print("You are using this VCTK+DEMAND training data:", NOISY_DATABASE_TRAIN)
        else:
            print("You are using this VCTK+DEMAND validation data:", NOISY_DATABASE_VALID)
        self.noisy_database_train = sorted(librosa.util.find_files(NOISY_DATABASE_TRAIN, ext='wav'))[:num_data_tot]
        self.noisy_database_valid = sorted(librosa.util.find_files(NOISY_DATABASE_VALID, ext='wav'))
        self.L = int(length_in_seconds * fs)
        self.random_start_point = random_start_point
        self.fs = fs
        self.length_in_seconds = length_in_seconds
        self.num_data_per_epoch = num_data_per_epoch
        self.train = train
        
    def sample_data_per_epoch(self):
        self.noisy_data_train = random.sample(self.noisy_database_train, self.num_data_per_epoch)

    def pad_or_trim(self, noisy, clean):
        audio_length = min(len(noisy), len(clean))
        if audio_length > self.L and self.random_start_point:
            start = np.random.randint(0, audio_length - self.L + 1)
        else:
            start = 0

        def process(audio):
            if len(audio) > self.L:
                audio = audio[start:start + self.L]
            elif len(audio) < self.L:
                audio = np.pad(audio, (0, self.L - len(audio)), mode='constant')
            return audio.astype(np.float32)

        return process(noisy), process(clean)

    def __getitem__(self, idx):
        if self.train:
            noisy_list = self.noisy_data_train
        else:
            noisy_list = self.noisy_database_valid

        noisy, _ = sf.read(noisy_list[idx], dtype='float32')
        clean, _ = sf.read(noisy_list[idx].replace('noisy', 'clean'), dtype='float32')

        noisy, clean = self.pad_or_trim(noisy, clean)

        return noisy, clean

    def __len__(self):
        if self.train:
            return self.num_data_per_epoch
        else:
            return len(self.noisy_database_valid)


if __name__=='__main__':
    from tqdm import tqdm 
    from omegaconf import OmegaConf
    
    config = OmegaConf.load('configs/cfg_train.yaml')

        
    train_dataset = VoiceBank_Demand(**config['train_dataset'])
    train_dataloader = data.DataLoader(train_dataset, **config['train_dataloader'])
    train_dataloader.dataset.sample_data_per_epoch()

    validation_dataset = VoiceBank_Demand(**config['validation_dataset'])
    validation_dataloader = data.DataLoader(validation_dataset, **config['validation_dataloader'])

    print(len(train_dataloader), len(validation_dataloader))

    for noisy, clean in tqdm(train_dataloader):
        print(noisy.shape, clean.shape)
        break
        # pass

    for noisy, clean in tqdm(validation_dataloader):
        print(noisy.shape, clean.shape)
        break
        # pass
