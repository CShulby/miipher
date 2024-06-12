import concurrent.futures
import logging
import os
import shutil
import tempfile
from typing import Any

import hydra
import torchaudio  # type: ignore[import-untyped]
from lightning_vocoders.models.hifigan.xvector_lightning_module import (  # type: ignore[import-untyped, note]
    HiFiGANXvectorLightningModule,
)

from miipher.dataset.preprocess_for_infer import PreprocessForInfer  # type: ignore[import-untyped]
from miipher.lightning_module import MiipherLightningModule  # type: ignore[import-untyped]
import time

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s:%(message)s",
)
logger = logging.getLogger(__name__)


def initialize_models() -> (
    tuple[
        MiipherLightningModule,
        HiFiGANXvectorLightningModule,
        Any,
        PreprocessForInfer,
    ]
):
    """
    Initialize and load the necessary models for processing.

    Returns:
        tuple: Contains instances of the Miipher model,
               vocoder model, xvector model, and preprocessor.
    """
    miipher_path = "miipher_v2.ckpt"
    vocoder_path = "vocoder_finetuned_v2.ckpt"
    logger.info("Loading models...")
    miipher = MiipherLightningModule.load_from_checkpoint(
        miipher_path,
        map_location="cuda",
    )
    vocoder = HiFiGANXvectorLightningModule.load_from_checkpoint(
        vocoder_path,
        map_location="cuda",
    )
    xvector_model = hydra.utils.instantiate(
        vocoder.cfg.data.xvector.model,
        run_opts={"device": "cuda"},
    )
    preprocessor = PreprocessForInfer(miipher.cfg).to("cuda")
    preprocessor.cfg.preprocess.text2phone_model.is_cuda = True
    logger.info("Models loaded successfully.")
    return miipher, vocoder, xvector_model, preprocessor


def find_transcript_path(wav_path):
    """
    Find the path to the corresponding transcript file for a given wav file.

    Args:
        wav_path (str): Path to the wav file.

    Returns:
        str: Path to the transcript file if found, otherwise None.
    """
    base_path = wav_path.rsplit(".", 1)[0]  # Remove the file extension
    possible_paths = [
        base_path + ".txt",  # Direct match
        base_path.replace(
            os.path.basename(base_path), "whisper_" + os.path.basename(base_path)
        )
        + ".txt",  # Prefixed
    ]
    for path in possible_paths:
        if os.path.exists(path):
            return path
    return None


def to_cuda(d: dict) -> dict:
    for k, v in d.items():
        if getattr(v, "keys", None):
            d[k] = to_cuda(v)
        else:
            d[k] = v.to("cuda")
    return d


def process_file(
    wav_path,
    transcript_path,
    lang_code,
    output_dir,
    miipher,
    vocoder,
    xvector_model,
    preprocessor: PreprocessForInfer,
):
    """
    Process a single wav file, performing speech enhancement and saving the cleaned output.

    Args:
        wav_path (str): Path to the wav file.
        transcript_path (str): Path to the transcript file.
        lang_code (str): Language code for processing.
        output_dir (str): Directory to save the processed file.
        miipher (object): Miipher model instance.
        vocoder (object): Vocoder model instance.
        xvector_model (object): Xvector model instance.
        preprocessor (object): Preprocessor instance.

    Returns:
        str: Path to the cleaned wav file if successful, otherwise None.
    """
    if transcript_path is None:
        logger.info(f"Transcript not found for {wav_path}")
        return None

    try:
        with open(transcript_path, "r") as f:
            transcript = f.read().strip()

        logger.info(f"Processing file: {wav_path}")
        wav, sr = torchaudio.load(wav_path)
        wav = wav[0].unsqueeze(0)
        wav = wav.to("cuda")
        batch = preprocessor.process(
            "test",
            (wav.clone().detach(), sr),
            word_segmented_text=transcript,
            lang_code=lang_code,
        )

        batch = to_cuda(batch)

        (
            phone_feature,
            speaker_feature,
            degraded_ssl_feature,
            _,
        ) = miipher.feature_extractor(batch)
        cleaned_ssl_feature, _ = miipher(
            phone_feature.detach(),
            speaker_feature.detach(),
            degraded_ssl_feature.detach(),
        )
        vocoder_xvector = xvector_model.encode_batch(
            batch["degraded_wav_16k"].view(1, -1).detach()
        ).squeeze(1)
        cleaned_wav = vocoder.generator_forward(
            {
                "input_feature": cleaned_ssl_feature.to("cuda").detach(),
                "xvector": vocoder_xvector.to("cuda").detach(),
            }
        )[0].T

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmpfile:
            torchaudio.save(
                tmpfile.name,
                cleaned_wav.to("cpu").view(1, -1),
                sample_rate=22050,
                format="wav",
            )
            output_file_name = os.path.basename(wav_path).replace(
                ".wav", "_cleaned.wav"
            )
            new_path = os.path.join(output_dir, output_file_name)
            shutil.move(tmpfile.name, new_path)
            logger.info(f"Processed and saved cleaned audio to {new_path}")
            return new_path
    except Exception as e:
        logger.error(f"Error processing {wav_path}: {e}")
        return None


def process_list(wav_list_path, lang_code, models, log_file_path):
    """
    Process a list of wav files specified in a text file.

    Args:
        wav_list_path (str): Path to the text file containing wav file paths.
        lang_code (str): Language code for processing.
        models (tuple): Tuple containing model instances.
        log_file_path (str): Path to the log file for missing transcripts.

    Returns:
        list: List of paths to successfully processed and cleaned wav files.
    """
    miipher, vocoder, xvector_model, preprocessor = models
    with open(wav_list_path, "r") as f:
        files = [line.strip() for line in f]
    valid_files = []
    invalid_files = []

    for file in files:
        transcript_path = find_transcript_path(file)
        if transcript_path:
            valid_files.append((file, transcript_path))
        else:
            invalid_files.append(file)

    log_missing_transcripts(invalid_files, log_file_path)

    if not valid_files:
        logger.info("No valid audio and transcript pairs found.")
        return []

    output_files = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        futures = [
            executor.submit(
                process_file,
                wav_path,
                transcript_path,
                lang_code,
                os.path.dirname(wav_path),
                miipher,
                vocoder,
                xvector_model,
                preprocessor,
            )
            for wav_path, transcript_path in valid_files
        ]
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            if result:
                output_files.append(result)
    return output_files


def log_missing_transcripts(files, log_file_path):
    """
    Log the files for which transcripts were not found.

    Args:
        files (list): List of file paths for which transcripts were not found.
        log_file_path (str): Path to the log file.
    """
    with open(log_file_path, "w") as log_file:
        for file in files:
            log_file.write(f"Transcript not found for {file}\n")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Process audio files.")
    parser.add_argument(
        "--wav_list",
        type=str,
        default="wav_list",
        help="Text file containing paths to wav files and their corresponding transcriptions",
    )
    parser.add_argument(
        "--lang_code",
        type=str,
        default="eng-us",
        help='Language code, e.g., "eng-us" or "jpn"',
    )

    args = parser.parse_args()

    # Initialize models
    models = initialize_models()
    log_file_path = "transcript_not_found.log"

    # Process the list of files
    process_list(args.wav_list, args.lang_code, models, log_file_path)
