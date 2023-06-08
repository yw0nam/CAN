import subprocess, argparse
from glob import glob

def define_argparser():
    p = argparse.ArgumentParser(description = "Make wav files from mp4 files")
    p.add_argument('--root_dir', type=str, required=True)
    config = p.parse_args() 
    return config

def main(config):

    vid_pathes = glob(config.root_dir+'/*/*.mp4')
    vid_pathes.sort()
    
    for vid_path in vid_pathes:
        wav_path = vid_path.replace('mp4', 'wav')
        command = ("ffmpeg -y -i %s -ac 1 -vn -acodec pcm_s16le -ar 16000 %s" % (vid_path, wav_path))
        output = subprocess.call(command, shell=True, stdout=None)

if __name__ == "__main__":
    args = define_argparser()
    main(args)