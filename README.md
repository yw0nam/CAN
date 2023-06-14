# Multimodal_Emotion_Recognition

MultiModal Emotion Recognition using Cross modal Attention module and Contrastive loss


# Installation
## Experiment setting

- Linux
- Python 3.8.16
- PyTorch 1.13.1 and CUDA 11.7

a. Create a conda virtual environment and activate it.

```shell
conda create -n MER python=3.8
conda activate MER
```

b. Install PyTorch and torchvision following the [official instructions](https://pytorch.org/)

c. Clone this repository.

d. Install requirments.

```shell
pip install -r requirements.txt
```

e. Install DeepSpeed

First you need libaio-dev. please install by

```shell
sudo apt-get install libaio-dev
```

After this, install deepspeed by 

```shell
DS_BUILD_CPU_ADAM=1 DS_BUILD_FUSED_ADAM=1 DS_BUILD_UTILS=1 DS_BUILD_AIO=1 pip install deepspeed==0.9.0 --global-option="build_ext" --global-option="-j11" --no-cache-dir
```

Please check for detail installation [DeepSpeed official github](https://github.com/microsoft/DeepSpeed)

# Train 

Run Training code

```shell
bash train_hf.sh
```

Check your GPU, and change train_hf.sh and configs properly.

# Tensorboard 

You can run tensorboard 

```shell
tensorboard --logdir ./output/log/tensorboard_what_you_want/version_0/
```

# Inference

Because this repository use deepspeed stage 2, model weights sharded between gpus. So you need to make sharded checkpoints as one.
You need to collate the model weights using

```shell
python make_model_weights.py
```

After this,

```shell
CUDA_VISIBLE_DEVICES=0 python inference.py
```
