# Image Segmentation with TensorFlow

This project demonstrates how to build and train neural network models for image segmentation using TensorFlow and Keras. The primary goal is to classify every pixel in an image (e.g., a cardiac MRI scan) into one of two classes (e.g., part of the left ventricle or not).

## Overview

In image segmentation, each pixel of an image is labeled, enabling the separation of regions of interest from the background. This project uses pre-processed medical images stored in TFRecords format and provides a modular codebase that supports three model architectures:

- **Task 1:** A Fully Connected Network (FCN) with one hidden layer.
- **Task 2:** A Convolutional Neural Network (CNN).
- **Task 3:** A CNN with an additional Dice coefficient metric for improved evaluation in imbalanced datasets.

## Features

- **Modular Design:**  
  The project is divided into functions for data loading, preprocessing, model building, training, and visualization.

- **Data Pipeline:**  
  Efficient data loading using TensorFlow's `tf.data` API with caching, shuffling, batching, and prefetching.

- **Multiple Models:**  
  Three model architectures to experiment with:
  - **Fully Connected (Task 1)**
  - **Convolutional (Task 2)**
  - **Convolutional with Dice Metric (Task 3)**

- **Visualization:**  
  Helper functions display input images, ground-truth labels, and model predictions side-by-side.
![Screenshot 2025-03-11 203054](https://github.com/user-attachments/assets/4336791b-fe2e-4e00-8f98-646405f6fba6)

![Screenshot 2025-03-11 203821](https://github.com/user-attachments/assets/5009ed1f-64a4-42d5-8bf3-52916dd36b45)


- **TensorBoard Integration:**  
  Logs training metrics to TensorBoard for real-time monitoring.

- **Command-Line Interface:**  
  Use `argparse` to select the task (model) to train with the `--task` parameter.

## Prerequisites

- **Python:** 3.6+
- **TensorFlow:** 2.4.0 or higher
- **Matplotlib:** 3.3.0 or higher

See the `requirements.txt` file for a complete list of dependencies.

## Installation

1. **Clone the repository:**

    ```bash
    git clone https://github.com/InfiniteLoopster-coder/Image-Segmentation.git
    cd Image-Segmentation
    ```

2. **Install dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

## Data Setup

Ensure that your TFRecords files for training and validation images are placed in a `data/` directory or update the file paths in the script accordingly:

- `TRAIN_TFRECORD = 'data/train_images.tfrecords'`
- `VAL_TFRECORD = 'data/val_images.tfrecords'`

## Usage

Run the script from the command line by specifying the task:

- **Task 1 (Fully Connected Network):**

    ```bash
    python Solution.py --task 1
    ```

- **Task 2 (Convolutional Neural Network):**

    ```bash
    python Solution.py --task 2
    ```

- **Task 3 (CNN with Dice Metric):**

    ```bash
    python Solution.py --task 3
    ```

> **Note:**  
> If running in a Jupyter Notebook, use `parse_known_args()` in the argument parser to ignore extra parameters that Jupyter might inject.

## TensorBoard

To monitor training progress, launch TensorBoard by running:

```bash
tensorboard --logdir logs
-**
