DIR="$1"
CKPT="$2"
PY_ARGS=${@:3}

python train.py --config "$DIR/hparams.yaml" --eval-path "$DIR/checkpoints/$CKPT"  DATASET.DATAROOT data/nuscenes N_WORKERS 10 ${PY_ARGS}
