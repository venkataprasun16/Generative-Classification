# Generative Digit Classification & Missing Feature Reconstruction

[![Dataset: MNIST](https://img.shields.io/badge/Dataset-MNIST-orange.svg)](https://yann.lecun.com/exdb/mnist/)

This project explores **generative machine learning models** for handwritten digit classification and reconstruction using the MNIST dataset.  
It focuses on Gaussian models, Gaussian Mixture Models, Expectation-Maximization, and missing pixel reconstruction using conditional probability.<br>
This project repository was developed as part of the **CS771: Introduction to Machine Learning** course at the **Indian Institute of Technology Kanpur (IITK)** under **Professor Purushottam Kar**.

---

## Project Overview

Instead of directly learning decision boundaries between digit classes, this project models the probability distribution of each digit class:

$$
P(X \mid Y)
$$

Using these learned distributions, the model can classify digits, discover handwriting styles, and reconstruct missing parts of digit images.

---

## Mathematical Concepts Involved

### 1. Generative Classification

Each digit class is modeled using a multivariate Gaussian distribution.

$$
P(X \mid Y = c)
$$

A test image is classified by choosing the digit class with the highest likelihood.

---

### 2. Multivariate Gaussian Distribution

A digit image is treated as a high-dimensional vector.  
The Gaussian model uses a mean vector and covariance matrix to capture the structure of pixel values.

$$
X \sim \mathcal{N}(\mu, \Sigma)
$$

---

### 3. Gaussian Mixture Model

A single Gaussian may not represent all handwriting styles well.  
A Gaussian Mixture Model represents data using multiple Gaussian components.

$$
P(X) = \sum_{k=1}^{K} \pi_k \mathcal{N}(X \mid \mu_k, \Sigma_k)
$$

---

### 4. Expectation-Maximization Algorithm

EM is used to estimate the parameters of a Gaussian Mixture Model.

- **E-Step:** Estimate the probability that each data point belongs to each Gaussian component.
- **M-Step:** Update the mean, covariance, and mixture weights using those probabilities.

---

### 5. Missing Feature Reconstruction

When some pixels are removed from an image, conditional Gaussian distributions are used to estimate the missing values.

$$
P(X_{\text{missing}} \mid X_{\text{observed}})
$$

This allows the model to reconstruct censored or incomplete digit images.

---

## Experiments Performed

### 1. Fitting a Single Gaussian Model

In this experiment, a single multivariate Gaussian distribution was fitted to digit images.  
Each image was treated as a high-dimensional vector, and the model estimated a mean vector and covariance matrix for the data.

![Single Gaussian](images/single_gaussian.png)

**Conclusion:**  
A single Gaussian gives a rough average representation of the digit data, but it fails to capture different handwriting styles. Since MNIST digits have large variation in shape, thickness, and orientation, one Gaussian is not expressive enough.

---

### 2. Fitting Multiple Gaussians

To better model the variation in handwritten digits, multiple Gaussian distributions were used.  
Each Gaussian component captures a different style or pattern present in the digit images.

![Multiple Gaussians](images/multiple_gaussians.png)

**Conclusion:**  
Multiple Gaussians represent the data better than a single Gaussian. They can capture different modes in the dataset, such as slanted digits, thick digits, thin digits, and different writing styles.

---

### 3. Gaussian Mixture Model using EM Algorithm

A Gaussian Mixture Model was trained using the Expectation-Maximization algorithm.  
The EM algorithm alternates between assigning soft cluster probabilities and updating Gaussian parameters.

![GMM Components](images/gmm_components.png)

**Conclusion:**  
The GMM successfully discovers hidden patterns in the data without using explicit style labels. The learned components show that generative models can separate different handwriting styles in an unsupervised way.

---

### 4. Generative Digit Classification

A generative classifier was built by fitting probability distributions for each digit class.  
For a test image, the model calculates the likelihood under each digit class and predicts the class with the highest probability.

![Classification Results](images/classification_results.png)

**Conclusion:**  
The generative classifier is able to classify MNIST digits by modeling class-wise data distributions. Although it may not perform as well as modern neural networks, it provides a strong probabilistic interpretation of classification.

---

### 5. Missing Pixel Censoring

In this experiment, a part of each digit image was removed or censored.  
The goal was to test whether the model could use the visible pixels to infer the missing region.

![Censored Digits](images/censored_digits.png)

**Conclusion:**  
Censoring important pixels makes classification harder because the central region often contains key digit structure. This experiment shows how missing features can reduce the amount of useful information available to the model.

---

### 6. Missing Feature Reconstruction

The missing pixels were reconstructed using conditional Gaussian distributions.  
The model estimated the missing region based on the observed pixels and the learned covariance structure.

![Reconstructed Digits](images/reconstructed_digits.png)

**Conclusion:**  
Conditional Gaussian reconstruction can recover meaningful digit shapes from incomplete images. The reconstructed digits are not always perfect, but they preserve important structural information and show the usefulness of probabilistic modeling.

---

## Repository Structure

```text
ML_Digit_classification/
├── lec15.ipynb
├── handwritten_digits_generative_modelling.ipynb
├── cs771/
│   ├── __init__.py
│   ├── plotData.py
│   └── utils.py
├── mnist/
│   ├── train-images.idx3-ubyte
│   ├── train-labels.idx1-ubyte
│   ├── t10k-images.idx3-ubyte
│   └── t10k-labels.idx1-ubyte
├── images/
│   ├── single_gaussian.png
│   ├── multiple_gaussians.png
│   ├── gmm_components.png
│   ├── classification_results.png
│   ├── censored_digits.png
│   └── reconstructed_digits.png
└── README.md
```

---

## Requirements

Install the required Python libraries:

```bash
pip install numpy matplotlib scikit-learn jupyter
```

---

## How to Run

Clone the repository:

```bash
git clone https://github.com/your-username/ML_Digit_classification.git
cd ML_Digit_classification
```

Start Jupyter Notebook:

```bash
jupyter notebook
```

Open and run:

```text
lec15.ipynb
handwritten_digits_generative_modelling.ipynb
```

---

## Results

- A single Gaussian model provided a basic average representation of digit data.
- Multiple Gaussians captured more variation in handwritten digit styles.
- GMM with EM discovered hidden digit patterns without explicit labels.
- The generative classifier classified digits using probability distributions.
- Missing pixel reconstruction recovered useful digit structure from incomplete images.
- The project demonstrates how probabilistic models can be used for both classification and reconstruction.

---

## Acknowledgement
This project repository was developed using MNIST dataset as part of the **CS771: Introduction to Machine Learning** course at the **Indian Institute of Technology Kanpur (IITK)** under **Professor Purushottam Kar**.
