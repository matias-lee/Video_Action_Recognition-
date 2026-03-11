# Video Action Recognition with UCF50 using LRCN

This repository implements a robust video classification pipeline for the **UCF50** dataset using a Long-term Recurrent Convolutional Network (LRCN). The architecture leverages a pre-trained ResNet34 backbone to extract spatial features from individual frames and an LSTM to model temporal dynamics across the video sequence.

A critical focus of this implementation is the strict prevention of data leakage. Videos in UCF50 are grouped by actors and environments. This pipeline utilizes a `GroupShuffleSplit` strategy to mathematically guarantee that clips from the same group do not cross the train/test boundaries, ensuring an honest evaluation of the model's ability to generalize temporal actions.

---

## Table of Contents
1. [Environment Setup](#environment-setup)
2. [Data Preparation & Preprocessing](#data-preparation--preprocessing)
3. [Training the Model](#training-the-model)
4. [Testing and Evaluation Metrics](#testing-and-evaluation-metrics)
5. [Repository Structure](#repository-structure)

---

## Environment Setup

This project requires **Python 3.10+** and a CUDA-enabled GPU for optimal training speed. 

To avoid C++ linking errors (such as `iJIT_NotifyEvent` with Intel MKL) on HPC clusters, it is highly recommended to install PyTorch via `pip` wheels within a Conda environment.

**1. Create and activate a clean Conda environment:**
`conda create -n video_action python=3.10 -y`
`conda activate video_action`

**2. Install PyTorch with statically linked CUDA 11.8 binaries:**
`pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118`

**3. Install required data processing and plotting libraries:**
`pip install opencv-python scikit-learn tqdm numpy Pillow matplotlib seaborn pylint`

**4. Install unrar (required to extract the UCF50 dataset):**
`conda install -c conda-forge unrar -y`

---


## Data Preparation & Preprocessing

### 1. Download the Dataset
We recommend storing the massive video files outside of the Git repository to prevent storage limits.

`mkdir -p ./data`
`cd ./data`
`wget -c --no-check-certificate "https://www.crcv.ucf.edu/data/UCF50.rar"`
`unrar x UCF50.rar`
`cd ..`

### 2. Extract Frames (Uniform Random Sampling)
Deep learning vision models require discrete image tensors. Run the preprocessing script to iterate through the raw `.avi` files, divide them into temporal segments, and apply **uniform random sampling** to extract exactly 16 frames per video.

`python preprocess.py`

*Note: Ensure your `RAW_DIR` and `OUT_DIR` paths in `preprocess.py` correctly point to your extracted UCF50 data.*

---


## Training the Model

The training pipeline automatically handles the `GroupShuffleSplit`, applies data augmentation (Random Horizontal Flips, Random Affine transformations), normalizes the tensors to ImageNet standards, and executes the training loop using a `ReduceLROnPlateau` scheduler.

To launch the training script on your GPU cluster:

`python run.py --frame_dir ./data/UCF50_frames --n_classes 50 --batch_size 8 --model_type lrcn --cnn_backbone resnet34 --mode train`

The script will automatically save the best performing model weights to `./models/best_model_wts.pt` and output the dataset splits to `./splits.npy` to guarantee reproducible evaluations.

---


## Testing and Evaluation Metrics

The evaluation module computes comprehensive multiclass metrics, bypassing simple accuracy to provide a robust assessment of the model across all 50 action categories.

To run the test suite using your saved model weights:

`python run.py --frame_dir ./data/UCF50_frames --n_classes 50 --batch_size 8 --ckpt ./models/best_model_wts.pt --mode eval`

**Evaluation Outputs:**
1. **Overall Test Accuracy:** Top-1 classification accuracy.
2. **Macro F1 Score:** Unweighted mean of the F1 scores for each class (robust to class imbalances).
3. **Macro AUC Score (OvR):** Area Under the ROC Curve computed using the One-vs-Rest strategy and softmax probabilities.
4. **Confusion Matrix Heatmap:** A high-resolution 50x50 visual matrix (`ucf50_confusion_matrix.png`) is automatically generated and saved to the root directory for inclusion in experimental writeups.

---

## Repository Structure
* `run.py`: Main execution script for training and evaluation.
* `train.py`: Contains the core backpropagation and validation loops.
* `test.py`: Houses the evaluation loop, multiclass metrics (AUC, F1), and confusion matrix plotter.
* `models.py`: Defines the LRCN PyTorch architecture (ResNet34 + LSTM).
* `video_datasets.py`: Custom PyTorch `Dataset` and `GroupShuffleSplit` dataloaders.
* `utils.py`: Transform pipelines and stochastic uniform frame sampling algorithms.
* `preprocess.py`: Automated raw video to sequence-tensor conversion script.


