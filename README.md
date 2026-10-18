# Rock-Paper-Scissors Hand Gesture Classification

A PyTorch image classification project that trains a small convolutional neural network (CNN) to recognize **paper**, **rock**, and **scissors** hand gestures.

The [notebook](Rock-Paper-Scissors.ipynb) covers image exploration, preprocessing, dataset splitting, model construction, and training and evaluation. The TinyVGG model achieves **90.10% test accuracy after 5 epochs** on the project's test split.

## Project overview

- Explore sample images and visualize their transformations.
- Load class-labeled images with `torchvision.datasets.ImageFolder`.
- Resize images to **64 × 64** pixels and apply random horizontal flipping.
- Split the dataset into **80% training / 20% test** subsets.
- Train a custom **TinyVGG** model from scratch using PyTorch.
- Report loss and accuracy for both subsets after each epoch.
- Use CUDA when available, otherwise CPU.

## Dataset

The project uses **Rock-Paper-Scissors Images** by **Julien de la Bruère-Terreault (DrGFreeman)**, available on [Kaggle](https://www.kaggle.com/datasets/drgfreeman/rockpaperscissors). The dataset is licensed under [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/) and was created for the author's [original computer vision project](https://github.com/DrGFreeman/rps-cv).

The dataset contains RGB PNG images measuring **300 × 200 pixels**, photographed against a green background with relatively consistent lighting and white balance.

The project uses **2,188 images** across three classes: paper, rock, and scissors. An 80/20 split provides **1,750 training images** and **438 test images**.

The expected extracted image directories, relative to the notebook's working directory, are:

- `data/rock_paper_scissors/rps-cv-images/paper/`
- `data/rock_paper_scissors/rps-cv-images/rock/`
- `data/rock_paper_scissors/rps-cv-images/scissors/`

**Dataset setup:** download the data from the Kaggle page before running the notebook. Check the downloaded archive's folder structure before running. The Colab extraction cell expects a file named `rps-cv-images.zip` containing the `rps-cv-images` directory and its three class folders. If the download has a different layout, manually extract the three class folders into the paths above.

## Model architecture

The model is a compact VGG-style CNN named `TinyVGG`, with **10,693 trainable parameters**.

| Stage | Layers | Output per image |
| --- | --- | --- |
| Input | RGB image | `3 × 64 × 64` |
| Convolution block 1 | Conv2d → ReLU → Conv2d → ReLU → MaxPool2d | `10 × 32 × 32` |
| Convolution block 2 | Conv2d → ReLU → Conv2d → ReLU → MaxPool2d | `10 × 16 × 16` |
| Classifier | Flatten → Linear | 3 class logits |

All convolution layers use 3 × 3 kernels with padding of 1. Each pooling layer reduces the spatial dimensions by half. The linear layer expects the feature dimensions produced by 64 × 64 input images.

## Training configuration

| Setting | Value |
| --- | --- |
| Input size | 64 × 64 RGB |
| Batch size | 32 |
| Epochs | 5 |
| Optimizer | Adam |
| Learning rate | 0.001 |
| Loss function | CrossEntropyLoss |
| Hidden channels | 10 |
| Class mapping | Paper: 0, Rock: 1, Scissors: 2 |

## Results

Test accuracy increases from **32.28%** to **90.10%** over five epochs, while test loss decreases from **1.0878** to **0.3258**.

| Epoch | Training loss | Training accuracy | Test loss | Test accuracy |
| --- | ---: | ---: | ---: | ---: |
| 1 | 1.0964 | 36.18% | 1.0878 | 32.28% |
| 2 | 0.9843 | 50.42% | 0.8644 | 69.38% |
| 3 | 0.7733 | 70.44% | 0.6130 | 79.38% |
| 4 | 0.6100 | 77.54% | 0.4537 | 84.86% |
| 5 | 0.4030 | 86.79% | 0.3258 | **90.10%** |

Training time: **32.512 seconds**. Runtime depends on the execution environment.

Loss and accuracy are averaged across batches, with evaluation performed after each epoch.

## How to run

### Option 1: Google Colab

The notebook contains a Google Drive archive path, so Colab is the most direct setup.

1. Download `Rock-Paper-Scissors.ipynb` and upload it to [Google Colab](https://colab.research.google.com/).
2. In Google Drive, place the matching dataset archive at:
   `MyDrive/Deep Learning/rps-cv-images.zip`.
3. Mount Google Drive using Colab's Files panel. The archive must be accessible at:
   `/content/drive/MyDrive/Deep Learning/rps-cv-images.zip`.
4. Optionally select a GPU runtime. The notebook automatically selects CUDA when available.
5. Run the cells from top to bottom.

The archive should contain an `rps-cv-images` directory with the three class folders inside it. Extraction runs only if `data/rock_paper_scissors` does not already exist; an empty or incomplete directory will not trigger re-extraction.

### Option 2: Local Jupyter environment

1. Download the notebook and [requirements.txt](requirements.txt) into the same project folder.
2. Create and activate a Python virtual environment.
3. From the project folder, install the dependencies listed in `requirements.txt`:

   ```bash
   python -m pip install -r requirements.txt
   ```

4. Manually extract the dataset into the expected directories listed above. This allows the notebook to skip its Colab-specific ZIP extraction step.
5. Start Jupyter from the project folder:

   ```bash
   jupyter notebook
   ```

6. Open `Rock-Paper-Scissors.ipynb` and run the cells in order.

For a GPU-specific PyTorch installation, use the instructions on the [official PyTorch installation page](https://pytorch.org/get-started/locally/).

## References

Learning resources used for this project:

- [Learn PyTorch — Custom Datasets](https://www.learnpytorch.io/04_pytorch_custom_datasets/)
- [PyTorch tutorial on YouTube](https://www.youtube.com/watch?v=Z_ikDlimN6A)
- [GeeksforGeeks — How to Split a Dataset Using PyTorch](https://www.geeksforgeeks.org/deep-learning/how-to-split-a-dataset-using-pytorch/)

