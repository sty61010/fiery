# 3DCV Final Project

This project is forked from https://github.com/wayveai/fiery.

## Environments

1. git clone this repo and create the necessary directorys
    ```bash
    https://github.com/sty61010/fiery.git
    cd fiery/
    mkdir -p data tensorboard_logs
    ```
    
2. Download NuScenes dataset and place it under the `data` directory.

2. Create the python environment from `bev18.yml`

    ```bash
    conda env create -n fiery --file bev18.yml
    ```

3. Install `mmdetection3d` package. For more information, please refer to [mmdet3d document](https://github.com/open-mmlab/mmdetection3d/blob/master/docs/en/getting_started.md).

    ```bash
    git clone https://github.com/open-mmlab/mmdetection3d.git
    cd mmdetection3d
    pip install -v -e .
    ```

## How to test the model

Download the pretrained weight here and extract them into `tensorboard_logs`.

In the `fiery` directory, run

```
bash test.sh tensorboard_logs/model 20.ckpt
```

It will run evaluation on the trainval dataset and report the mAP score.

Expected result:


![image-20220603205905601](/home/andy94077/fiery/README.assets/image-20220603205905601.png)

## How to train the model

There are many configs in the `fiery/config` folder, you can train the model by

```bash
python train.py --config CONFIG GPUS "[0]" ...[Other Parameters]
```

To preproduce our pretrained model, run

```bash
python train.py --config tensorboard_logs/model/hparams.yml
```