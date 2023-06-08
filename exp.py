# %%
import pandas as pd
import numpy as np
# %%
csv = pd.read_csv('./data/MELD/train.csv')
# %%
csv['wav_length']
# %%
csv.query(f"wav_length >=1.5 and wav_length < 12")
# %%
csv.query("wav_length <= 1.0")
# %%
