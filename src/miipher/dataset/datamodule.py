from lightning.pytorch import LightningDataModule
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
import webdataset as wds
import torch
import torchaudio
import hydra
from transformers import AutoFeatureExtractor


class MiipherDataModule(LightningDataModule):
    def __init__(self, cfg) -> None:
        super().__init__()

        self.speech_ssl_processor = hydra.utils.instantiate(
            cfg.data.speech_ssl_processor.processor
        )

        #self.speech_ssl_processor = processor = AutoFeatureExtractor.from_pretrained("facebook/w2v-bert-2.0")
        self.speech_ssl_sr = cfg.data.speech_ssl_processor.sr
        #self.phoneme_tokenizer = hydra.utils.instantiate(cfg.data.phoneme_tokenizer)
        self.cfg = cfg

    def setup(self, stage: str):
        self.train_dataset = (
            wds.WebDataset(
                self.cfg.data.train_dataset_path,
                resampled=True,
                nodesplitter=wds.split_by_node,
            )
            .shuffle(1000)
            .decode(wds.torch_audio)
            # .decode(self.decode_phoneme_input)
            .repeat(2)
            .with_length(20000 * self.cfg.data.train_batch_size)
        )
        self.val_dataset = (
            wds.WebDataset(
                self.cfg.data.val_dataset_path, nodesplitter=wds.split_by_node
            )
            .decode(wds.torch_audio)
            # .decode(self.decode_phoneme_input)
            .repeat(2)
            .with_length(3000 * 4 // self.cfg.data.val_batch_size)
        )

    def train_dataloader(self):
        print("Initializing training DataLoader")
        dataloader = DataLoader(
            self.train_dataset,
            batch_size=self.cfg.data.train_batch_size,
            collate_fn=self.collate_fn,
            num_workers=8,
        )
        print(f"Training DataLoader created with batch size: {self.cfg.data.train_batch_size}")
        return dataloader

    def val_dataloader(self):
        print("Initializing validation DataLoader")
        dataloader = DataLoader(
            self.val_dataset,
            batch_size=self.cfg.data.val_batch_size,
            collate_fn=self.collate_fn,
            num_workers=8,
        )
        print(f"Validation DataLoader created with batch size: {self.cfg.data.val_batch_size}")
        return dataloader


    def custom_padding(self, batch_texts):
        # Find the maximum length of text in the batch
        max_length = max(len(text) for text in batch_texts)
        # Initialize a matrix to store the padded texts
        padded_texts = torch.zeros((len(batch_texts), max_length), dtype=torch.long)
        # Fill the matrix with the batch texts with padding
        for i, text in enumerate(batch_texts):
            padded_texts[i, :len(text)] = torch.tensor(text)
        return padded_texts
    
    @torch.no_grad()
    def collate_fn(self, batch):
        print("Starting collate_fn...")
        output = dict()
        degraded_wav_16ks = []
        clean_wav_16ks = []

        for sample in batch:
            #print("Processing a sample...")
            clean_wav, sr = sample["speech.wav"]
            clean_wav_16ks.append(
                torchaudio.functional.resample(clean_wav, sr, new_freq=16000).squeeze()[:16000*20]
            )
            degraded_wav, sr = sample["degraded_speech.wav"]
            degraded_wav_16ks.append(
                torchaudio.functional.resample(
                    degraded_wav, sr, new_freq=16000
                ).squeeze()[:16000*20]
            )
        output["degraded_wav_16k"] = pad_sequence(degraded_wav_16ks, batch_first=True)
        output["degraded_wav_16k_lengths"] = torch.tensor(
            [degraded_wav_16k.size(0) for degraded_wav_16k in degraded_wav_16ks]
        )
        output["clean_ssl_input"] = self.speech_ssl_processor(
            [x.numpy() for x in clean_wav_16ks],
            return_tensors="pt",
            sampling_rate=16000,
            padding=True,
        )
        output["degraded_ssl_input"] = self.speech_ssl_processor(
            [x.numpy() for x in degraded_wav_16ks],
            return_tensors="pt",
            sampling_rate=16000,
            padding=True,
        )
        #output["phoneme_input_ids"] = self.phoneme_tokenizer(
        #    [b["phoneme.txt"] for b in batch], return_tensors="pt", padding=True
        #)
        #batch_texts = [b["phoneme_input_ids.pth"] for b in batch]
        #output["phoneme_input_ids"] = self.custom_padding(batch_texts)
        print("Collecting phoneme input IDs...")
        try:
            batch_texts = [b["phoneme_input_ids.pth"] for b in batch]
            #print("Batch texts collected:", batch_texts)
            output["phoneme_input_ids"] = self.custom_padding(batch_texts)
            #print("Phoneme input IDs padded and added to output.")
        except KeyError as e:
            print(f"Key error during batch processing: {e}")

        print("Batch processing completed.")
        return output
