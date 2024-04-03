from typing import Any
from torch import nn as nn
import torchaudio
import random
import pyroomacoustics as pra
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm
import subprocess
import tempfile
import os


class DegrationApplier:
    def __init__(self, cfg) -> None:
        self.format_encoding_pairs = cfg.format_encoding_pairs
        self.reverb_conditions = cfg.reverb_conditions
        self.background_noise = cfg.background_noise
        self.cfg = cfg
        self.rirs = []
        self.prepare_rir(cfg.n_rirs)
        self.noise_audio_paths = []

        for root, pattern in self.cfg.background_noise.patterns:
            print(f"Searching for paths in {root} with pattern {pattern}:")
            path_list = list(Path(root).rglob(pattern))  # Use rglob for recursive globbing
            self.noise_audio_paths.extend(path_list)

            if path_list:  # If paths were found
                print(f"Found {len(path_list)} paths:")
                for p in path_list[:5]:  # Print the first 5 paths found
                    print(p, "exists." if p.exists() else "does not exist.")
            else:
                print("No paths found.")

    def applyCodec(self, waveform, sample_rate):
        # Assuming all encoding tasks should leverage FFmpeg for this example
        try:
            with tempfile.NamedTemporaryFile(suffix='.wav', mode='w+b', delete=True) as input_tmp_wav, \
                 tempfile.NamedTemporaryFile(suffix='.mp3', mode='w+b', delete=False) as output_tmp_mp3:

                # Save waveform to temporary WAV file
                torchaudio.save(input_tmp_wav.name, waveform, sample_rate)
                input_tmp_wav.flush()

                # Convert WAV to MP3 using FFmpeg
                subprocess.run(['ffmpeg', '-y', '-i', input_tmp_wav.name, output_tmp_mp3.name], check=True)

                # Optionally, load MP3 back into Python if needed
                waveform, sample_rate = torchaudio.load(output_tmp_mp3.name, format='mp3')
        except subprocess.CalledProcessError as e:
            print(f"Error during audio codec application: {e}")
            # Handle the error or re-raise as appropriate
        finally:
            # Cleanup if any temporary file was created
            if output_tmp_mp3:
                output_tmp_mp3.close()
                #os.remove(output_tmp_mp3.name)

        return waveform

        param = random.choice(self.format_encoding_pairs)
        waveform = torchaudio.functional.apply_codec(
            waveform=waveform.float(), sample_rate=sample_rate, **param
        )
        return waveform.float()

    def applyReverb(self, waveform):
        if len(self.rirs) == 0:
            raise RuntimeError
        rir = random.choice(self.rirs)
        augmented = torchaudio.functional.fftconvolve(waveform, rir)
        return augmented.float()

    def prepare_rir(self, n_rirs):
        for i in tqdm(range(n_rirs)):
            xy_minmax = self.reverb_conditions.room_xy
            z_minmax = self.reverb_conditions.room_z
            x = random.uniform(xy_minmax.min, xy_minmax.max)
            y = random.uniform(xy_minmax.min, xy_minmax.max)
            z = random.uniform(z_minmax.min, z_minmax.max)
            corners = np.array([[0, 0], [0, y], [x, y], [x, 0]]).T
            room = pra.Room.from_corners(corners, **self.reverb_conditions.room_params)
            room.extrude(z)
            room.add_source(self.cfg.reverb_conditions.source_pos)
            room.add_microphone(self.cfg.reverb_conditions.mic_pos)

            room.compute_rir()
            rir = torch.tensor(np.array(room.rir[0]))
            rir = rir / rir.norm(p=2)
            self.rirs.append(rir)

    def applyBackgroundNoise(self, waveform, sample_rate):
        snr_max, snr_min = self.background_noise.snr.max, self.background_noise.snr.min
        snr = random.uniform(snr_min, snr_max)
        print(f"Number of noise paths available: {len(self.noise_audio_paths)}")
        if len(self.noise_audio_paths) == 0:
            print("Warning: No noise audio paths found. Please check the configuration and path loading.")
        else:
            sample_paths = self.noise_audio_paths[:5]  # Adjust the number of paths as needed
            print(f"Sample available noise paths: {sample_paths}")
        noise_path = random.choice(self.noise_audio_paths)
        print(f"Applying noise from: {noise_path}") 
        noise, noise_sr = torchaudio.load(noise_path)
        noise /= noise.norm(p=2)
        if noise.size(0) > 1:
            noise = noise[0].unsqueeze(0)
        noise = torchaudio.functional.resample(noise, noise_sr, sample_rate)
        if not noise.size(1) < waveform.size(1):
            start_idx = random.randint(0, noise.size(1) - waveform.size(1))
            end_idx = start_idx + waveform.size(1)
            noise = noise[:, start_idx:end_idx]
        else:
            noise = noise.repeat(1, waveform.size(1) // noise.size(1) + 1)[
                :, : waveform.size(1)
            ]
        augmented = torchaudio.functional.add_noise(
            waveform=waveform, noise=noise, snr=torch.tensor([snr])
        )
        return augmented

    def process(self, waveform, sample_rate):
        if len(waveform.shape) == 1:
            waveform = waveform.unsqueeze(0)
        waveform = self.applyBackgroundNoise(waveform, sample_rate)
        if random.random() > self.cfg.reverb_conditions.p:
            waveform = self.applyReverb(waveform)
        waveform = self.applyCodec(waveform, sample_rate)
        return waveform.squeeze()

    def __call__(self, waveform, sample_rate):
        return self.process(waveform, sample_rate)
