#!/bin/bash
# Runs the commands to train model

set -e

usage="$(basename "$0") [-h] [-l LEARNING_RATE] [-e NUM_EPOCHS] [-b BATCH_SIZE] [-i IMAGE_SIZE]
Runs training and evaluation of an encoder-decoder image colorization model.
    -h  show this help message
    -l  learning rate
    -e  number of epochs to train
    -b  batch size
    -i  image size to resize to"

LEARNING_RATE=0.01
NUM_EPOCHS=60
BATCH_SIZE=50
IMAGE_SIZE=64

while getopts ":hl:e:b:i:" opt; do
    case "$opt" in
        h)
            echo "$usage"
            exit 0
            ;;
        l)
            LEARNING_RATE="$OPTARG"
            ;;
        e)
            NUM_EPOCHS="$OPTARG"
            ;;
        b)
            BATCH_SIZE="$OPTARG"
            ;;
        i)
            IMAGE_SIZE="$OPTARG"
            ;;
        :)
            echo "Error: Option -$OPTARG requires an argument."
            echo "$usage"
            exit 1
            ;;
        \?)
            echo "Error: Invalid option -$OPTARG"
            echo "$usage"
            exit 1
        ;;
    esac
done

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"
echo "Running from: $REPO_ROOT"

echo "======================== Prepare ========================"
echo "Calculating chroma..."
python3 -u ./code/chroma.py

echo "Calculating bins..."
python3 -u ./code/bins.py

echo "======================== TRAIN ========================"
python3 -u ./code/train.py --learning_rate $LEARNING_RATE --num_epochs $NUM_EPOCHS --batch_size $BATCH_SIZE --image_size $IMAGE_SIZE

echo -e "\n\n======================== EVALUATE ========================"
python3 -u ./code/evaluate.py --model_path data/output/best_model-$LEARNING_RATE-$NUM_EPOCHS-$BATCH_SIZE.pt --image_size $IMAGE_SIZE

echo -e "\n\n======================== DONE ========================"
