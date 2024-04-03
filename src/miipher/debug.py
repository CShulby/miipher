import webdataset as wds
import torchaudio
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, filename='dataset_debug.log', filemode='w')
logger = logging.getLogger(__name__)

# Specify your dataset paths
train_dataset_path = "/home/christopher/Corpora/LibriTTS-R/dev_clean.tar.gz"  # Update this to your dataset path
val_dataset_path = "/home/christopher/Corpora/LibriTTS-R/test_clean.tar.gz"      # Update this to your dataset path

def main():
    # Load the dataset
    train_dataset = wds.WebDataset(train_dataset_path).decode(wds.torch_audio)

    # Iterate over the dataset
    for i, sample in enumerate(train_dataset):
        if i >= 10:  # Inspect only the first 10 samples
            break

        # Check for expected keys in the sample
        keys_present = sample.keys()
        logger.info(f"Sample {i} keys: {keys_present}")

        if "speech.wav" not in sample:
            logger.warning(f"Missing 'speech.wav' in sample {i}")
        else:
            logger.info(f"'speech.wav' found in sample {i}")

        if "degraded_speech.wav" not in sample:
            logger.warning(f"Missing 'degraded_speech.wav' in sample {i}")
        else:
            logger.info(f"'degraded_speech.wav' found in sample {i}")
            
        if 'wav' in sample:
            waveform, sample_rate = sample['wav']
            print(f"Sample {i} waveform shape: {waveform.shape}, Sample Rate: {sample_rate}")
            # Optional: Play the audio to verify it's correct
            # torchaudio.play(waveform, sample_rate)


if __name__ == "__main__":
    main()

