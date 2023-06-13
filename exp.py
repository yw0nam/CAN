# %%|
from models.pl_model_hf import PL_model
from omegaconf import OmegaConf as OC
import pandas as pd
import numpy as np
from transformers import AutoTokenizer, Wav2Vec2Processor
from transformers import AutoModel, Wav2Vec2ForCTC
from transformers import ViTImageProcessor, ViTModel
from dataset_hf import multimodal_dataset, multimodal_collator
from torch.utils.data import DataLoader
from torch import nn
from moviepy.editor import VideoFileClip
import torch
# %%
vid_path = "/data/research_data/dataset/MELD.Raw/train_splits/dia0_utt0.mp4"
clip = VideoFileClip(vid_path)
# %%
frames = []
for frame in clip.iter_frames():
    frames.append(frame)
# %%
csv = pd.read_csv('./data/MELD/train.csv')
# %%
processor = ViTImageProcessor.from_pretrained('facebook/dino-vitb16')
model = ViTModel.from_pretrained('facebook/dino-vitb16')
# %%
vit_input = processor(frames[0], return_tensor='pt')
# %%
wav2vec = Wav2Vec2ForCTC.from_pretrained("ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition")
# %%
wav2vec.lm_head = nn.Linear(1024, 768)
# %%
text_tokenizer = AutoTokenizer.from_pretrained("roberta-base")
audio_processor = Wav2Vec2Processor.from_pretrained("jonatasgrosman/wav2vec2-large-xlsr-53-english")

train_dataset = multimodal_dataset(csv)
# %%
train_dataset
# %%
train_loader = DataLoader(
    train_dataset, 16, num_workers=4,
    collate_fn=multimodal_collator(text_tokenizer, audio_processor), pin_memory=True,
    shuffle=True, drop_last=True
)

# %%
for i in train_loader:
    text_inputs, audio_inputs, labels = i
    break
# %%
config = OC.load('./configs/MELD/train.yaml')
model = PL_model(config)

# %%
emo_out, sim, contrastive_label = model.forward(text_inputs, audio_inputs)
# %%
outs = wav2vec(**audio_inputs)
# %%
