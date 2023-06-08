import os
import pandas as pd
import argparse

def define_argparser():
    p = argparse.ArgumentParser(description = "Make csv file for dataset")
    p.add_argument('--wav_root', type=str, required=True)
    p.add_argument('--csv_path', type=str, required=True)
    p.add_argument('--out_path', type=str, required=True)
    config = p.parse_args() 
    return config

def main(config):
    csv = pd.read_csv(config.csv_path)
    csv['Utterance'] = csv['Utterance'].map(lambda x: x.replace("", '\''))
    csv['Utterance'] = csv['Utterance'].map(lambda x: x.replace("", '.'))
    csv['wav_path'] = csv.apply(
        lambda x: os.path.join(
            config.wav_root, 
            "dia{}_utt{}.wav".format(x['Dialogue_ID'], x["Utterance_ID"])
    ), axis=1)
    csv = csv[csv['wav_path'].map(lambda x: os.path.exists(x))]
    csv[['Utterance','Emotion', 'wav_path']].to_csv(config.out_path,index=False)

if __name__ == '__main__':
    args = define_argparser()
    main(args)
