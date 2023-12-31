import numpy as np
import argparse
import torch

meld_emotion2int = {
    'neutral': 0, 
    'anger' : 1,
    'joy' : 2, 
    'surprise' : 3, 
    'sadness' : 4, 
    'fear' : 5, 
    'disgust' : 6
}

iemocap_emotion2int = {
    'Neutral': 0,
    'Frustration': 1,
    'Anger': 2,
    'Excited': 3,
    'Sadness': 4,
    'Happiness': 5
}
def pad_mel(inputs):
    _pad = 0

    def _pad_one(x, max_len):
        mel_len = x.shape[0]
        return np.pad(x, [[0, max_len - mel_len], [0, 0]], mode='constant', constant_values=_pad)

    max_len = max((x.shape[0] for x in inputs))
    return np.stack([_pad_one(x, max_len) for x in inputs])

def pad_data(x, length):
    _pad = 0
    return np.pad(x, (0, length - x.shape[0]), mode='constant', constant_values=_pad)

def prepare_data(inputs):
    max_len = max((len(x) for x in inputs))
    return np.stack([pad_data(x, max_len) for x in inputs])


def get_sinusoid_encoding_table(n_position, d_hid, padding_idx=None):
    ''' Sinusoid position encoding table '''

    def cal_angle(position, hid_idx):
        return position / np.power(10000, 2 * (hid_idx // 2) / d_hid)

    def get_posi_angle_vec(position):
        return [cal_angle(position, hid_j) for hid_j in range(d_hid)]

    sinusoid_table = np.array([get_posi_angle_vec(pos_i)
                              for pos_i in range(n_position)])

    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

    if padding_idx is not None:
        # zero vector for padding dimension
        sinusoid_table[padding_idx] = 0.

    return torch.FloatTensor(sinusoid_table)


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
    
def make_normalized_emotion(x, emo_dict):
    emo_arr = np.zeros(len(emo_dict))
    for emo in x.split(';'):
        if emo == '' or emo not in emo_dict.keys():
            continue
        emo_arr[emo_dict[emo]] += 1
    return emo_arr / emo_arr.sum()