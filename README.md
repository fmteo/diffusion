# Denoising Diffusion Probabilistic Models 

This project trains and compares a simple CNN and a U-Net for DDPM-based image denoising. It also provides a Streamlit application for viewing denoising results.

## Project files

- `DiffusionCNN.py` — defines the timestep-conditioned CNN.
- `DiffusionUNet.py` — defines the timestep-conditioned U-Net.
- `train.py` — selects the model and dataset, trains and evaluates the model, saves a checkpoint, and generates an MSE graph.
- `app.py` — Streamlit application for uploading images and viewing denoising results.
- `data/` — dataset files.
- `models/` — saved model checkpoints.
- `results/` — generated training graphs.
- `requirements.txt` — Python dependencies.
- `Dockerfile` — instructions for packaging the application environment.
- `test_diffusion.py` — automated tests for the models and diffusion schedule.

## Setup

Clone the repository:

```bash
git clone <repository-url>
cd diffusion
```

Create and activate a Python environment, then install the dependencies:

```bash
python -m venv .venv
```

On Windows:

```bash
.venv\Scripts\activate
```

On macOS or Linux:

```bash
source .venv/bin/activate
```

Install the required packages:

```bash
pip install -r requirements.txt
```

## Train with MNIST

MNIST is downloaded automatically by `torchvision` if it is not already present in the data directory.

Train the CNN:

```bash
python train.py --dataset mnist --model cnn
```

Train the U-Net:

```bash
python train.py --dataset mnist --model unet
```

The default settings use 20 epochs, a batch size of 128, a learning rate of `2e-4`, and 1,000 diffusion timesteps. Override settings with command-line arguments, for example:

```bash
python train.py \
  --dataset mnist \
  --model cnn \
  --epochs 20 \
  --batch-size 64 \
  --lr 2e-4 \
  --width 64 \
  --T 1000 \
  --beta-start 0.0001 \
  --beta-end 0.02 \
  --val-fraction 0.1 \
  --seed 123
```

## Train with an image-folder dataset

Place the images under a single directory. The loader searches recursively for supported image formats. For the colour-image experiment, images should be RGB and have consistent dimensions; resize them to 128 × 128 before training if that is the intended model input size.

```bash
python train.py \
  --dataset folders \
  --model unet \
  --data-dir ./data/faces \
  --epochs 20 \
  --batch-size 4 \
  --width 64 \
  --seed 123
```

For folder datasets, the script creates the test split from the supplied image directory, then creates the validation split from the remaining training data. With 3,000 images and the default 10% validation fraction, this produces 2,430 training images, 270 validation images, and 300 test images.

Reduce the batch size if training exceeds available memory. Use the same dataset, model settings, and seed when reproducing an experiment.

## Outputs

The training script saves model checkpoints under `models/` and MSE graphs under `results/`. The checkpoints include the model weights and key diffusion settings needed for evaluation.

The reported training, validation, and test MSE values measure **noise-prediction error**. They do not directly measure pixel-level error between a final reconstructed image and its clean source.

## Run the Streamlit application

After training, ensure `app.py` is configured to load the desired checkpoint, then start the application:

```bash
streamlit run app.py
```

Open the local URL printed by Streamlit in a browser.

## Run with Docker

Build the Docker image:

```bash
docker build -t diffusion-denoising .
```

Run the container:

```bash
docker run --rm -p 8501:8501 diffusion-denoising
```

Open `http://localhost:8501` in a browser. The Dockerfile must start Streamlit on `0.0.0.0` and expose port `8501` for this command to work.

## Run automated tests

Run the tests from the project directory:

```bash
python -m pytest -v test_diffusion.py
```

The tests check the diffusion schedule, forward-noise calculation, model output shapes, a small training step, and checkpoint evaluation. They check software behavior; they do not establish denoising quality.

## Reproducibility

The training script accepts command-line options for the model, dataset, training parameters, diffusion schedule, and random seed. It sets seeds for the data split and training process and saves key model and schedule settings with each checkpoint. To reproduce a run, use the same code revision, dataset, command-line settings, and checkpoint configuration.