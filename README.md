# AI Image Detector using PyTorch and EfficientNet-B7

![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)
![Kaggle](https://img.shields.io/badge/Kaggle-20BEFF?style=for-the-badge&logo=Kaggle&logoColor=white)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)

A deep learning project designed to accurately classify images as either **Real** or **AI-Generated**. This solution uses a fine-tuned **EfficientNet-B0** model, implemented in PyTorch, to achieve high accuracy on this binary classification task.

The model is trained on the "AI Generated Images vs Real Images" dataset from Kaggle to distinguish between authentic photographs and synthetic images from various AI models.

***

## 📖 Table of Contents
* [About The Project](#about-the-project)
* [Model Architecture](#model-architecture-🔬)
* [Dataset](#dataset-🖼️)
* [Getting Started](#getting-started-🚀)
    * [Prerequisites](#prerequisites)
    * [Installation](#installation)
* [Usage](#usage-💡)
* [Results](#results-📊)
* [Contributing](#contributing-🤝)
* [License](#license-📜)

---

## About The Project

As AI image generation becomes more advanced, the need for reliable detection tools is growing. This project provides a powerful and accessible solution by leveraging a state-of-the-art convolutional neural network (CNN) to perform this classification.

The core of this detector is **EfficientNet-B0**, a highly efficient and accurate model, which has been fine-tuned using a comprehensive set of data augmentations to ensure robustness and generalization.


---

## Model Architecture 🔬

This project utilizes **EfficientNet-B0**, the baseline model from the EfficientNet family developed by Google AI. EfficientNet models are known for their exceptional balance of accuracy and computational efficiency, achieved through a novel **compound scaling method**. This method uniformly scales network width, depth, and resolution with a fixed set of coefficients.

EfficientNet-B0 was chosen for its lightweight nature and fast performance, making it ideal for applications where resource efficiency is important. The model was used as a feature extractor with a custom classifier head added for the binary classification task.

---

## Dataset 🖼️

The model was trained on the **AI Generated Images vs Real Images** dataset, publicly available on Kaggle. This dataset provides a collection of images split into two classes:
* **AI-Generated (`fake`):** Images created by various AI models.
* **Real (`real`):** Authentic photographs.

A stratified 80/20 split was used to create the training and testing sets, ensuring that the class distribution was maintained in both.

**Data Augmentation:** To improve model robustness, the following augmentations were applied to the training data:
* Random Horizontal Flips
* Random Rotations
* Color Jitter (brightness, contrast, saturation, hue)

---

## Getting Started 🚀

Follow these instructions to get a local copy of the project up and running for development and testing purposes.

### Prerequisites

You will need Python 3.8+ and pip installed. The project was developed in a Kaggle environment and uses libraries like PyTorch, TorchMetrics, and Scikit-learn.

### Installation

1.  **Clone the repository:**
    ```sh
    git clone [https://github.com/your-username/ai-image-detector.git](https://github.com/your-username/ai-image-detector.git)
    cd ai-image-detector
    ```

2.  **Create a virtual environment (recommended):**
    ```sh
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install the required packages:**
    A `requirements.txt` file should be created with the following key dependencies:
    ```
    numpy
    torch
    torchvision
    torchmetrics
    Pillow
    scikit-learn
    pandas
    matplotlib
    seaborn
    kaggle
    ```
    Install them using pip:
    ```sh
    pip install -r requirements.txt
    ```

4.  **Download the Dataset:**
    The notebook uses the `kagglehub` library to download the dataset. Ensure you have your Kaggle API token configured.
    ```python
    import kagglehub
    path = kagglehub.dataset_download("cashbowman/ai-generated-images-vs-real-images")
    ```

---

## Usage 💡

The core logic for training and evaluation is contained within the `Project.ipynb` notebook. To use the project:

1.  **Open the Jupyter Notebook:**
    Launch Jupyter Notebook and open `Project.ipynb`.

2.  **Run the Cells:**
    Execute the cells in order to:
    * Download and prepare the dataset.
    * Define the data transformations and create DataLoaders.
    * Initialize the EfficientNet-B0 model, optimizer, and scheduler.
    * Run the training loop for the desired number of epochs (the notebook runs for 50 epochs).
    * Evaluate the model performance and visualize the confusion matrix.

3.  **Prediction on a New Image:**
    To classify a new image, you can adapt the prediction logic from the evaluation loop. You'll need to load the trained model weights, apply the `test_transform` to your image, and pass it to the model.

---

## Results 📊

The model's performance is evaluated using accuracy and a confusion matrix. The training loop tracks both training and validation accuracy across epochs, adjusting the learning rate with `ReduceLROnPlateau` based on validation performance.

The final output of the notebook includes a plot of the confusion matrix, providing a clear visual representation of the model's ability to distinguish between the two classes.

| Metric         | Value  |
| :------------- | :----- |
| **Accuracy** | *87%* |
| **Optimizer** | Adam   |
| **Image Size** | 224x224|


---

## Contributing 🤝

Contributions are what make the open-source community an amazing place to learn, inspire, and create. Any contributions you make are **greatly appreciated**.

1.  Fork the Project
2.  Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3.  Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4.  Push to the Branch (`git push origin feature/AmazingFeature`)
5.  Open a Pull Request

---

## License 📜

Distributed under the MIT License. See `LICENSE.txt` for more information.
