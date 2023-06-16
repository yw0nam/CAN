from torch.utils.data import Dataset
import torch
import librosa
from utils import *

class multimodal_dataset(Dataset):
    
    def __init__(self, csv, config, sort=True):
        self.csv = csv
        self.dataset_name = config['exp_setting']['dataset_name']
        self.graylabel = config['exp_setting']['using_graylabel']
        if sort:
            self.csv = self.csv.sort_values(by=['wav_length'])
    def __len__(self):
        return len(self.csv)
    
    def _load_data(self, idx):
        
        wav, _ = librosa.load(self.csv['wav_path'].iloc[idx], sr=16000)
        txt = self.csv['text'].iloc[idx]
        
        if not self.graylabel:
            emotion = self.csv['emotion'].iloc[idx]
            emotion = meld_emotion2int[emotion] if self.dataset_name == 'MELD' else iemocap_emotion2int[emotion]
        else:
            emotion = self.csv['full_emotion'].iloc[idx]
            emotion = make_normalized_emotion(emotion, emo_dict=iemocap_emotion2int)
        sample = {
            'text' : txt,
            'wav' : wav,
            'emotion': emotion
        }
        return sample
    
    def __getitem__(self, idx):
        sample = self._load_data(idx)
        return sample

class multimodal_collator():
    
    def __init__(self, text_tokenizer, audio_processor, return_text=False, max_length=512, graylabel=False):
        self.text_tokenizer = text_tokenizer
        self.audio_processor = audio_processor
        self.return_text = return_text
        self.max_length = max_length
        self.graylabel = graylabel
    def __call__(self, batch):
        text = [d['text'] for d in batch]
        wav = [d['wav'] for d in batch]
        emotion = [d['emotion'] for d in batch]
        
        text_inputs = self.text_tokenizer(
            text,
            padding=True,
            truncation=True,
            return_tensors="pt",
            add_special_tokens=True,
            max_length=self.max_length
        )
        
        audio_inputs = self.audio_processor(
            wav,
            sampling_rate=16000, 
            padding=True, 
            return_tensors='pt'
        )
        
        labels = {
            "emotion" : torch.LongTensor(emotion) if not self.graylabel else torch.FloatTensor(emotion)
        }
        if self.return_text:
            labels['text'] = text
        return text_inputs, audio_inputs, labels