import argparse
import string
from transformers import AutoTokenizer
#from text_normalize import normalize_text, remove_accents
from miipher.ml_pl_bert.text_normalize import normalize_text, remove_accents
from miipher.ml_pl_bert.text_utils import TextCleaner
import phonemizer

class PhonemizeProcessor:
    special_mappings = {
        "a": "ɐ",
        "'t": 't',
        "'ve": "v",
        "'m": "m",
        "'re": "ɹ",
        "d": "d",
        'll': "l",
        "n't": "nt",
        "'ll": "l",
        "'d": "d",
        "'": "ʔ",
        "wasn": "wˈɒzən",
        "hasn": "hˈæzn",
        "doesn": "dˈʌzən",
    }

    def __init__(self, language='en-us', preserve_punctuation=True, with_stress=True):
        self.global_phonemizer = phonemizer.backend.EspeakBackend(
            language=language,
            preserve_punctuation=preserve_punctuation,
            with_stress=with_stress
        )
        self.text_cleaner = TextCleaner() 
        self.tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-multilingual-cased")


    def _phonemize(self, text):
        text = normalize_text(remove_accents(text))
        words = self.tokenizer.tokenize(text)
        
        phonemes_bad = [self.global_phonemizer.phonemize([word], strip=True)[0] if word not in string.punctuation else word for word in words]
        input_ids = []
        phonemes = []
        
        for i in range(len(words)):
            word = words[i]
            phoneme = phonemes_bad[i]
            
            for k, v in self.special_mappings.items():
                if word == k:
                    phoneme = v
                    break
            
            # Process special cases
            if word == "'s" and i > 0 and phonemes[i - 1][-1] in ['s', 'ʃ', 'n']:
                phoneme = "z" if phonemes[i - 1][-1] in ['s', 'ʃ', 'n'] else "s"
            
            if i != len(words) - 1 and word in ["haven", "don"] and words[i+1] == "'t":
                phoneme = "hˈævn" if word == "haven" else "dˈəʊn"
            
            if word == "the" and i < len(words) - 1:
                next_phoneme = phonemes_bad[i + 1].replace('ˈ', '').replace('ˌ', '')
                if next_phoneme[0] in 'ɪiʊuɔɛeəɜoæʌɑaɐ':
                    phoneme = "ðɪ"
            
            if word == "&" and 0 < i < len(words):
                phoneme = "ænd"
            
            if word == "A" and i > 0 and words[i - 1] == ".":
                phoneme = "ɐ"
            
            if "@" in word and len(word) > 1:
                phoneme = word.replace('@', '')
                input_ids.append(self.tokenizer.encode(phoneme)[1:-1])
                continue
            
            input_ids.append(self.tokenizer.encode(word)[1:-1])
            phonemes.append(phoneme)
            
        
        assert len(input_ids) == len(phonemes)
        phonemes_str = ' '.join(phonemes)#.encode('utf-8')
        phonemes_str = ''.join(phonemes_str)
        return phonemes_str
    

    def __call__(self, text):
        phonemes = self._phonemize(text)
        input_ids = self.text_cleaner(phonemes)
        return input_ids, phonemes
        #return {'input_ids': input_ids, 'phonemes': phonemes}
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pl_bert_dir", default="src/miipher/ml_pl_bert/checkpoints")

    args = parser.parse_args()

    input_text = "It is not in the stars to hold our destiny but in ourselves; we are underlings."
    preprocessor = PhonemizeProcessor(language='en-us')
    input_ids, phonemes = preprocessor(input_text)

    print(input_text)
    print(phonemes)
    print(input_ids)
