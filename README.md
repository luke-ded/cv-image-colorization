# Image Colorization as Quantized Color Classification

This project creates an encoder-decoder image colorization model from scratch with a U-Net style architecture. The goal of the model is to take a grayscale input (L) and produce a color output (ab). The model’s output is quantized over ab color bins to produce vivid colors instead of the muddier results produced by models without this step. Muddiness is also mitigated through class rebalancing; rare, vivid colors are weighted more heavily than common colors. The model is trained on a subset of the Places365 dataset; only selected outdoor classes were used for this
project. The model created in this project produces colorized results that typically look reasonable based on visual inspection. Furthermore, the model produced a 23.33 dB PSNR versus a 21.94 dB for the grayscale input. This is a notable result; beating the PSNR value of grayscale input means the model’s color guesses are right more often than they are confidently wrong, resulting in plausible looking predictions. Colorfulness and visual inspection are used for evaluation in addition to PSNR/SSIM. Notably, a model without class rebalancing scores best on PSNR/SSIM (25.38 dB) but is duller than real photos (colorfulness 32.06 vs. 42.73). In the training process, random-crop augmentation gave the largest gain in these metrics.

## Setup and Run Instructions

### Requirements

- Python 3 (3.10-3.14) and bash
- Packages in `scripts/requirements.txt`: torch, torchvision, numpy, scikit-image, matplotlib
- About 55 GB of free disk space for the Places365 download

### Setup

```bash
bash scripts/setup_venv.bash
source .venv/bin/activate
```

### Run

```bash
bash scripts/run.bash [-l LEARNING_RATE] [-e NUM_EPOCHS] [-b BATCH_SIZE] [-i IMAGE_SIZE]
```

The defaults are the final model's settings: `-l 0.01 -e 60 -b 50 -i 64`. Pass the learning rate as a plain decimal.

The script runs four steps in order:

1. `chroma.py` downloads Places365 to `data/` on the first run and computes chroma for the grayscale filter.
2. `bins.py` computes the color bins and rebalancing weights.
3. `train.py` trains the model.
4. `evaluate.py` reports PSNR, SSIM and colorfulness on the test set.

Training uses CUDA, MPS or the CPU, whichever is available. A full run takes about 3.5 hours on an Apple M5 Pro GPU.

Outputs are written to `data/output/`: the best checkpoint, a per-epoch loss CSV and sample images.

To evaluate an existing checkpoint:

```bash
python3 code/evaluate.py --model_path data/output/best_model-0.01-60-50.pt
```

`data/chroma.pt` and `data/bins.pt` must already be created to run the evaluation script.

## Report

![Report page 1](report/page_images/page-01.png)

![Report page 2](report/page_images/page-02.png)

![Report page 3](report/page_images/page-03.png)

![Report page 4](report/page_images/page-04.png)

![Report page 5](report/page_images/page-05.png)

![Report page 6](report/page_images/page-06.png)

![Report page 7](report/page_images/page-07.png)

![Report page 8](report/page_images/page-08.png)

![Report page 9](report/page_images/page-09.png)

![Report page 10](report/page_images/page-10.png)

![Report page 11](report/page_images/page-11.png)

![Report page 12](report/page_images/page-12.png)

![Report page 13](report/page_images/page-13.png)

![Report page 14](report/page_images/page-14.png)

![Report page 15](report/page_images/page-15.png)

