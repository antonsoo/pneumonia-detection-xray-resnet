# Pneumonia Detection from Chest X-Rays using Fine-Tuned ResNet

This project demonstrates the use of deep learning, specifically transfer learning with a fine-tuned ResNet model, to detect pneumonia from chest X-ray images. The model is trained on the [Chest X-Ray Images (Pneumonia) dataset](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia) from Kaggle.

-------

## Project Description

The goal of this project is to develop a computer vision model that can accurately classify chest X-ray images as either "normal" or "pneumonia." This is a binary classification problem that has significant implications for medical diagnosis and healthcare.

The project utilizes a pre-trained ResNet-50 model, a deep convolutional neural network architecture that has achieved state-of-the-art results on various image recognition tasks. By leveraging transfer learning, we can adapt the pre-trained model to our specific task and dataset, achieving high accuracy with relatively little training data.

-------

## Dataset

The dataset used in this project is the Chest X-Ray Images (Pneumonia) dataset available on Kaggle. It consists of 5,863 chest X-ray images labeled as either "normal" or "pneumonia."

**Dataset Structure:**

```plaintext
chest-xray/
├── train/
│   ├── NORMAL/
│   └── PNEUMONIA/
├── val/
│   ├── NORMAL/
│   └── PNEUMONIA/
└── test/
    ├── NORMAL/
    └── PNEUMONIA/
```

**Data Source:** [https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)

-------

## Data Preprocessing and Augmentation

The following data preprocessing and augmentation techniques were applied to the images:

*   **Resizing:** Images were resized to 256x256 pixels.
*   **Center Cropping:** Images were center-cropped to 224x224 pixels.
*   **Random Resized Cropping:** Images were randomly resized and cropped for training.
*   **Random Horizontal Flipping:** Images were randomly flipped horizontally for training.
*   **Normalization:** Pixel values were normalized using the ImageNet mean and standard deviation.

These augmentations help to improve the model's ability to generalize to unseen data and prevent overfitting.

## Model Architecture

The model architecture is based on ResNet-50, a deep convolutional neural network with 50 layers. The model was pre-trained on the ImageNet dataset, which contains millions of images from 1,000 different classes.

**Modifications for Fine-tuning:**

*   The final fully connected layer of the ResNet-50 model was replaced with a new fully connected layer that has 2 output neurons (for "normal" and "pneumonia").
*   The last few layers of the ResNet-50 model were unfrozen, allowing their weights to be updated during training.

-------

## Training Process

The model was trained using the following parameters:

*   **Optimizer:** Adam
*   **Learning Rate:** 0.001 (with a learning rate scheduler that decays the learning rate by a factor of 0.1 every 7 epochs)
*   **Batch Size:** 64
*   **Number of Epochs:** 25 (adjust as needed)
*   **Loss Function:** Cross-Entropy Loss

-------

## Evaluation Results

The model achieved the following performance on the test set:

*   **Accuracy:** \[Insert your test accuracy here]
*   **Precision:** \[Insert your test precision here]
*   **Recall:** \[Insert your test recall here]
*   **F1-Score:** \[Insert your test F1-score here]
*   **AUC:** \[Insert your test AUC here]

**Confusion Matrix:**

\[Insert your confusion matrix here if you have it]

**Training and Validation Loss/Accuracy Curves:**

\[Insert plots of your training and validation loss/accuracy curves here]

-------

## Installation

1. Clone this repository:

    ```bash
    git clone https://github.com/antonsoo/pneumonia-detection-xray-resnet
    ```

2. Navigate to the project directory:

    ```bash
    cd pneumonia-detection-xray-resnet
    ```

3. Install the required packages:

    ```bash
    pip install -r requirements.txt
    ```

4. Download the dataset:

   ```bash
   cd src
   python download_data.py paultimothymooney/chest-xray-pneumonia ../data/chest_xray
   ```

-------

## Usage

Please use the Notebook located in the `notebook` directory, or use the following steps:

1. **Download the dataset:**
    *   Option 1: Download the dataset from Kaggle using the instructions in the "Dataset" section.
    *   Option 2: If you have already downloaded the dataset, place it in the `data` directory, following the structure mentioned above.
2. **Train the model:**
    *   Run the `train.py` script:
        ```bash
        python src/train.py
        ```
    *   This will train the model and save the trained weights to the `model` directory.
3. **Evaluate the model:**
    *   Run the `evaluate.py` script:
        ```bash
        python src/evaluate.py
        ```
    *   This will evaluate the trained model on the test set and print the results.

-------

## Demo

(Optional) If you have created a demo (e.g., using Streamlit or Gradio), provide a link to it here.

-------

## Ethical Considerations

*   This project uses a publicly available dataset for research and educational purposes.
*   The model is intended to be a tool to assist medical professionals and should not be used as a replacement for a doctor's diagnosis.
*   It is important to be aware of potential biases in medical datasets and to consider the ethical implications of deploying AI in healthcare.

-------

## Future Improvements

*   Explore other pre-trained models (e.g., Inception, DenseNet).
*   Experiment with more advanced data augmentation techniques.
*   Fine-tune more layers of the pre-trained model.
*   Incorporate techniques to address class imbalance if it exists in the dataset.
*   Develop a user-friendly web application for real-time pneumonia detection.

-------

## Author

Anton Soloviev - https://www.upwork.com/freelancers/~01b9d171164a005062

-------

## License

This project is licensed under the [MIT License] - see the [LICENSE](LICENSE) file for details.

-------

## Acknowledgements

*   [Kaggle](https://www.kaggle.com/) for providing the Chest X-Ray Images (Pneumonia) dataset.
*   [PyTorch](https://pytorch.org/) for the deep learning framework.
*   [torchvision](https://pytorch.org/vision/stable/index.html) for pre-trained models and image transformations.
