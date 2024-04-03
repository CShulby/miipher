import os
from miipher.dataset.preprocess_for_infer import PreprocessForInfer
from miipher.lightning_module import MiipherLightningModule
from lightning_vocoders.models.hifigan.xvector_lightning_module import HiFiGANXvectorLightningModule
import torch
import torchaudio
import hydra
import tempfile
import shutil

# Load the models outside of the main function
miipher_path = "miipher_v2.ckpt"
miipher = MiipherLightningModule.load_from_checkpoint(miipher_path, map_location='cpu')
vocoder = HiFiGANXvectorLightningModule.load_from_checkpoint("vocoder_finetuned_v2.ckpt", map_location='cpu')
xvector_model = hydra.utils.instantiate(vocoder.cfg.data.xvector.model)
xvector_model = xvector_model.to('cpu')
preprocessor = PreprocessForInfer(miipher.cfg)
preprocessor.cfg.preprocess.text2phone_model.is_cuda = False

def find_transcript_path(wav_path):
    """
    Attempts to find a matching transcript for the given wav_path.
    Looks for a direct match and a 'whisper_' prefixed version.
    Returns the path of the found transcript or None if not found.
    """
    base_path = wav_path.rsplit('.', 1)[0]  # Remove the file extension
    possible_paths = [
        base_path + '.txt',  # Direct match
        base_path.replace(os.path.basename(base_path), 'whisper_' + os.path.basename(base_path)) + '.txt',  # Prefixed
    ]
    for path in possible_paths:
        if os.path.exists(path):
            return path
    return None

@torch.inference_mode()
def process_directory(wav_dir, output_dir, lang_code):
    output_files = []
    for file in os.listdir(wav_dir):
        if file.endswith('.wav'):
            wav_path = os.path.join(wav_dir, file)
            transcript_path = find_transcript_path(wav_path)
            if transcript_path is None:
                print(f"Transcript not found for {wav_path}")
                continue
            
            with open(transcript_path, 'r') as f:
                transcript = f.read().strip()
            
            # Process each file
            output_file = process_file(wav_path, transcript, lang_code, output_dir)
            output_files.append(output_file)
    
    return output_files


def process_file(wav_path, transcript, lang_code, output_dir):
    wav, sr = torchaudio.load(wav_path)
    wav = wav[0].unsqueeze(0)
    batch = preprocessor.process(
        'test',
        (torch.tensor(wav), sr),
        word_segmented_text=transcript,
        lang_code=lang_code
    )

    (
        phone_feature,
        speaker_feature,
        degraded_ssl_feature,
        _,
    ) = miipher.feature_extractor(batch)
    cleaned_ssl_feature, _ = miipher(phone_feature, speaker_feature, degraded_ssl_feature)
    vocoder_xvector = xvector_model.encode_batch(batch['degraded_wav_16k'].view(1, -1).cpu()).squeeze(1)
    cleaned_wav = vocoder.generator_forward({"input_feature": cleaned_ssl_feature, "xvector": vocoder_xvector})[0].T
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmpfile:
        torchaudio.save(tmpfile.name, cleaned_wav.view(1, -1), sample_rate=22050, format='wav')
        # Use output_dir to construct new path
        output_file_name = os.path.basename(wav_path).replace('.wav', '_cleaned.wav')
        new_path = os.path.join(output_dir, output_file_name)
        shutil.move(tmpfile.name, new_path)
        return new_path


if __name__ == "__main__":
    # Directory containing wav files and their corresponding transcriptions
    wav_dir = 'test_wav/'
    output_dir = wav_dir
    lang_code = 'eng-us'  # Or 'jpn', depending on your data
    
    # Ensure the output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Process the directory
    process_directory(wav_dir, output_dir, lang_code)

