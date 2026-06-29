# Generative-Classification
This project helps in understanding Generative classification, Missing feature reconstruction using Gaussian models, Gaussian Mixture Models, Expectation Maximisation.
# Generative Classification on MNIST

## 1. Overview

This project studies **generative classification** using the MNIST handwritten digit dataset.

The main idea of generative classification is:

> Instead of directly learning a decision boundary between classes, we learn how each class generates data.

For every digit class, we model the distribution of images belonging to that class:

\[
P(x \mid y = c)
\]

Then, for a test image \(x\), we predict the class that most likely generated it:

\[
\hat{y}
=
\arg\max_c P(y = c \mid x)
\]

Using Bayes rule:

\[
P(y = c \mid x)
=
\frac{P(x \mid y = c)P(y = c)}{P(x)}
\]

Since \(P(x)\) is the same for all classes during comparison, we use:

\[
\hat{y}
=
\arg\max_c P(x \mid y = c)P(y = c)
\]

In log form:

\[
\hat{y}
=
\arg\max_c
\left[
\log P(x \mid y = c) + \log P(y = c)
\right]
\]

The notebook first explores Gaussian models for generating digit images, then uses them for classification and missing-pixel reconstruction.

---

## 2. Mathematical Concepts Used

### 2.1 Data Representation

Each MNIST image has size:

\[
28 \times 28
\]

Each image is flattened into a vector:

\[
x \in \mathbb{R}^{784}
\]

So each pixel becomes one coordinate of the feature vector.

Pixel values are normalized:

\[
x = \frac{x}{256}
\]

---

### 2.2 Mean Image

The **mean image** is the pixel-wise average of all images in a dataset.

If we have \(n\) images:

\[
x^1, x^2, \dots, x^n
\]

then the mean image is:

\[
\mu =
\frac{1}{n}
\sum_{i=1}^{n} x^i
\]

For each pixel position, we average that pixel across all training images.

If the dataset contains both `0`s and `4`s, the mean image looks like a blurry mixture of `0` and `4`.

In class-wise generative classification, for class \(c\), the mean is:

\[
\mu_c =
\frac{1}{n_c}
\sum_{i:y^i=c} x^i
\]

where:

- \(n_c\) is the number of training examples in class \(c\)
- \(x^i\) is a training image
- \(y^i\) is its label

So for digit `4`:

\[
\mu_4 =
\frac{1}{n_4}
\sum_{i:y^i=4} x^i
\]

This gives the average image of all `4`s.

---

### 2.3 Covariance Matrix

The covariance matrix tells us how different pixels vary together.

For a full dataset:

\[
\Sigma =
\frac{1}{n}
\sum_{i=1}^{n}
(x^i - \mu)(x^i - \mu)^T
\]

For class \(c\):

\[
\Sigma_c =
\frac{1}{n_c}
\sum_{i:y^i=c}
(x^i - \mu_c)(x^i - \mu_c)^T
\]

Here:

- \(\Sigma_c\) is the covariance matrix of class \(c\)
- \(\mu_c\) is the mean image of class \(c\)
- \(x^i - \mu_c\) measures how image \(i\) differs from the class mean

Since each MNIST image has 784 pixels:

\[
\Sigma_c \in \mathbb{R}^{784 \times 784}
\]

The diagonal entries show how much each pixel varies by itself.

The off-diagonal entries show how two different pixels vary together.

Example:

- If two nearby stroke pixels are often dark together, covariance is positive.
- If one region is dark when another region is usually light, covariance is negative.
- If two pixels are unrelated, covariance is close to zero.

This is important because digit images are not independent pixels. They are structured strokes.

---

### 2.4 Standard Gaussian

A standard Gaussian model uses identity covariance:

\[
x \sim \mathcal{N}(\mu, I)
\]

This assumes:

- each pixel is independent
- each pixel has variance 1
- different pixels have covariance 0

This is a very weak model for images because it cannot learn stroke structure.

In the notebook, this produces noisy random images.

---

### 2.5 Spherical Gaussian

A spherical Gaussian uses scaled identity covariance:

\[
x \sim \mathcal{N}(\mu, \sigma^2 I)
\]

This is slightly more flexible than identity covariance because the variance can be learned, but it still assumes pixels are independent.

In the `7` and inverted `7` experiment, the notebook first uses:

\[
\Sigma = \sigma^2 I
\]

This performs poorly because it cannot capture relationships between pixel regions.

---

### 2.6 Full Gaussian

A full Gaussian learns both the mean and the covariance matrix:

\[
x \sim \mathcal{N}(\mu, \Sigma)
\]

This is more powerful because it captures how pixels vary together.

The Gaussian density is:

\[
P(x)
=
\frac{1}
{\sqrt{(2\pi)^d |\Sigma|}}
\exp
\left(
-\frac{1}{2}
(x-\mu)^T
\Sigma^{-1}
(x-\mu)
\right)
\]

where:

- \(d = 784\)
- \(\mu\) is the mean image
- \(\Sigma\) is the covariance matrix

The term:

\[
(x-\mu)^T \Sigma^{-1} (x-\mu)
\]

is a Mahalanobis distance. It measures distance from the mean while considering covariance.

---

### 2.7 Class Prior Probability

The prior probability of class \(c\) is:

\[
P(y=c)
=
\frac{n_c}{n}
\]

where:

- \(n_c\) is the number of training examples in class \(c\)
- \(n\) is the total number of training examples

This tells us how common a class is in the dataset.

For example, if digit `1` appears often, then:

\[
P(y=1)
\]

will be relatively high.

---

### 2.8 Generative Classification Rule

For each class \(c\), we learn:

\[
P(x \mid y=c)
=
\mathcal{N}(x \mid \mu_c, \Sigma_c)
\]

Then we classify using:

\[
\hat{y}
=
\arg\max_c
P(x \mid y=c)P(y=c)
\]

In log form:

\[
\hat{y}
=
\arg\max_c
\left[
\log P(x \mid y=c)
+
\log P(y=c)
\right]
\]

For a Gaussian class conditional distribution:

\[
\log P(x \mid y=c)
=
-\frac{1}{2}
\log |\Sigma_c|
-
\frac{1}{2}
(x-\mu_c)^T
\Sigma_c^{-1}
(x-\mu_c)
+
\text{constant}
\]

The constant can be ignored during classification because it is the same for all classes.

So the class score becomes:

\[
\text{score}(c)
=
-\frac{1}{2}
\log |\Sigma_c|
-
\frac{1}{2}
(x-\mu_c)^T
\Sigma_c^{-1}
(x-\mu_c)
+
\log P(y=c)
\]

The predicted class is the one with the highest score.

---

### 2.9 Gaussian Mixture Model

A single Gaussian may not capture all variations in a dataset.

For example, digit `4` may have many writing styles:

- upright `4`
- slanted `4`
- open-top `4`
- closed-top `4`

A Gaussian Mixture Model uses multiple Gaussians:

\[
P(x)
=
\sum_{c=1}^{C}
P(z=c)P(x \mid z=c)
\]

where:

- \(C\) is the number of components
- \(z\) is a latent variable
- \(z=c\) means component \(c\) generated the point

Each component has its own mean and covariance:

\[
\mu_c, \Sigma_c
\]

---

### 2.10 Latent Variable

A latent variable is a hidden variable.

In a Gaussian mixture model:

\[
z_i
\]

represents which Gaussian component generated data point \(x^i\).

We do not observe \(z_i\), so we estimate it.

For example:

\[
z_i = 3
\]

means image \(i\) was generated by component 3.

---

### 2.11 Expectation Maximization Algorithm

EM is used when the model contains latent variables.

The goal is to maximize:

\[
\sum_{i=1}^{n}
\log
\left(
\sum_{c=1}^{C}
P(x^i, z_i=c \mid \theta)
\right)
\]

This is difficult because there is a log of a sum.

EM solves this by repeating two steps.

#### E-Step

Compute responsibilities:

\[
q_c^i
=
P(z_i = c \mid x^i, \theta^{old})
\]

This tells us how likely data point \(i\) belongs to component \(c\).

For each point:

\[
\sum_{c=1}^{C} q_c^i = 1
\]

#### M-Step

Update model parameters using the responsibilities as weights.

The mean update is:

\[
\mu_c
=
\frac{
\sum_{i=1}^{n} q_c^i x^i
}
{
\sum_{i=1}^{n} q_c^i
}
\]

The covariance update is:

\[
\Sigma_c
=
\frac{
\sum_{i=1}^{n}
q_c^i
(x^i-\mu_c)(x^i-\mu_c)^T
}
{
\sum_{i=1}^{n} q_c^i
}
\]

So EM is:

```text
Initialize means and covariances
E-step: compute soft assignments q
M-step: update means and covariances using q
Repeat