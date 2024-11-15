# ML_GANs: Monet-Style Image Generation

This repository contains the implementation for the Kaggle competition ["I'm Something of a Painter Myself"](https://www.kaggle.com/c/gan-getting-started), which focuses on generating Monet-style paintings using Generative Adversarial Networks (GANs).

## Overview
The goal of this project is to transform photos into paintings in the style of Claude Monet. We use GANs to create art that mimics Monet's characteristic impressionist style.

## Project Structure
```
ML_GANs/
│
├── logs/                      # Training logs directory
│
├── models/                    # Model implementations
│   ├── __init__.py
│   └── model files
│
├── utils/                     # Utility functions
│   ├── __init__.py
│   └── utility files
│
├── dataset/                   # Dataset directory (to be created)
│   ├── monet_jpg/            # Monet paintings
│   └── photo_jpg/            # Real world photos
│
├── checkpoints/              # Model checkpoints directory
├── generated_imgs/           # Generated images directory
├── main.py                    # Main training script
│
└── requirements.txt           # Project dependencies
```

## Requirements
To install the required packages:
```bash
pip install -r requirements.txt
```

## Dataset Setup
1. Create a `dataset` folder in the root directory
2. Download the competition dataset from [Kaggle](https://www.kaggle.com/c/gan-getting-started/data)
3. Extract and organize the data as follows:
   - Place Monet paintings in `dataset/monet_jpg/`
   - Place photos in `dataset/photo_jpg/`

## Usage

### Command Line Arguments
The training script supports various command-line arguments for customization:

```bash
python main.py [arguments]
```

Available arguments:
- `--data_dir`: Path to the dataset directory (default: 'datasets')
- `--epochs`: Number of training epochs (default: 30)
- `--batch_size`: Batch size for training (default: 4)
- `--lr`: Learning rate for optimizers (default: 0.0002)
- `--beta1`: Beta1 parameter for Adam optimizer (default: 0.5)
- `--beta2`: Beta2 parameter for Adam optimizer (default: 0.999)
- `--model_save_dir`: Directory to save model checkpoints (default: './checkpoints')
- `--img_save_dir`: Directory to save generated images (default: 'generated_imgs')
- `--log_save_dir`: Directory to save training logs (default: 'logs')
- `--device`: Device to use for training (default: 'cuda:0')
- `--lambda_cycle`: Weight for cycle consistency loss (default: 10.0)
- `--seed`: Random seed for reproducibility (default: 42)

### Example Commands

1. Basic training with default parameters:
```bash
python main.py
```

2. Training with custom parameters:
```bash
python main.py --epochs 50 --batch_size 8 --lr 0.0001 --device cuda:0
```

3. Training on CPU:
```bash
python main.py --device cpu
```

4. Custom directories:
```bash
python main.py --data_dir custom_dataset --model_save_dir custom_checkpoints --img_save_dir custom_images
```

## Training Details
- Training progress and metrics are automatically logged to the `logs` directory
- Each training run creates a timestamped log file (format: `training_YYYY-MM-DD_HH-MM-SS.log`)
- Model checkpoints are saved in the `checkpoints` directory
- Generated images during training are saved in the `generated_imgs` directory
- The training process uses:
  - Adam optimizer with configurable learning rate and beta parameters
  - Cycle consistency loss with configurable weight
  - Automatic device selection (CUDA if available, CPU as fallback)

## Model Architecture
The implementation uses a GAN architecture suitable for style transfer:
- Generator: Transforms photos into Monet-style paintings
- Discriminator: Distinguishes between real Monet paintings and generated images

## License
This project is licensed under the MIT License.

## Acknowledgments
- Kaggle competition: ["I'm Something of a Painter Myself"](https://www.kaggle.com/c/gan-getting-started)
- Dataset provided by the competition organizers