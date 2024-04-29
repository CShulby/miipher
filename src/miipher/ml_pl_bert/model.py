import argparse
import os
import torch
import yaml
from glob import glob
from tqdm import tqdm

from miipher.ml_pl_bert.text_utils import TextCleaner

from transformers import AutoTokenizer, AlbertConfig, AlbertModel

import torch
from torch import nn
import torch.nn.functional as F

os.environ['PHONEMIZER_ESPEAK_PATH'] = '/usr/bin/espeak-ng'

class MlPlBertModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.encoder = self._load_bert_model()
        #self.mask_predictor = nn.Linear(hidden_size, num_tokens)
        #self.word_predictor = nn.Linear(hidden_size, num_vocab)
    
    def forward(self, phonemes, attention_mask=None):
        #print("Phonemes {}".format(phonemes.shape))
        output = self.encoder(phonemes, attention_mask=attention_mask)
        #tokens_pred = self.mask_predictor(output.last_hidden_state)
        #words_pred = self.word_predictor(output.last_hidden_state)
        
        return output.last_hidden_state
    
    def _load_bert_model(self, pl_bert_dir="src/miipher/ml_pl_bert/checkpoints"):
        print(f"Loading PL-Bert from {pl_bert_dir}...")
        config_path = os.path.join(pl_bert_dir, "config.yml")
        plbert_config = yaml.safe_load(open(config_path))
        
        albert_base_configuration = AlbertConfig(**plbert_config['model_params'])
        bert = AlbertModel(albert_base_configuration)

        model_ckpt_path = os.path.join(pl_bert_dir,"step_1100000.t7")
        state_dict = torch.load(model_ckpt_path)

        new_state_dict = {}
        for k in state_dict['net'].keys():
            new_k = k.replace("module.encoder.", "")
            new_state_dict[new_k] = state_dict['net'][k]

        missing_keys, unexpected_keys = bert.load_state_dict(new_state_dict, strict=True)

        print("Total missing_keys: {}".format(len(missing_keys)))
        print("Total unexpected_keys: {}".format(len(unexpected_keys)))

        if (len(missing_keys) == 0) and (len(unexpected_keys) == 0):
            print("Model Loaded with Success!")

        return bert
    

if __name__ == "__main__":
    from miipher.ml_pl_bert.phonemize_processor import PhonemizeProcessor

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    parser = argparse.ArgumentParser()
    parser.add_argument("--pl_bert_dir", default="src/miipher/ml_pl_bert/checkpoints")

    args = parser.parse_args()

    input_text = "It is not in the stars to hold our destiny but in ourselves; we are underlings."
    print(input_text)
    phonemize_processor = PhonemizeProcessor(language='en-us')
    input_ids, phonemes = phonemize_processor(input_text)
    #input_ids, phonemes = result['input_ids'], result['phonemes']
    #input_phonemes = ''.join(phonemes)
    print(phonemes)
    print(input_ids)

    input_ids_pt = torch.tensor(input_ids, dtype=torch.int64).unsqueeze(0).to(device)
    model = MlPlBertModel().to(device)
    ml_pl_bert_pred = model(input_ids_pt)

    print(input_text)
    print(ml_pl_bert_pred)
    print(ml_pl_bert_pred.shape)
