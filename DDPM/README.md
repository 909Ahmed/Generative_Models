# **Denoising Diffusion Probabilistic Model (DDPM)**
### This project implements a basic version of the Denoising Diffusion Probabilistic Model (DDPM) trained on the MNIST dataset.The core of the DDPM framework lies in its use of a Markov chain process to denoise/noisify the images.

## Installation

### To run this project, you need Python installed on your system along with the following dependencies:
- TensorFlow
- NumPy
- Matplotlib

## Improvements

#### One issue faced during training is the problem of exploding gradients. This can occur when gradients become too large during backpropagation, leading to numerical instability and slow convergence.

#### I have tried gradient clipping to prevent the gradients getting too large and ofcourse also used BatchNormalization. Though it helped in training the model but the issue seems to persist if trained for more epochs

## Contributing
#### Contributions to this project are welcome! If you have any suggestions, bug reports, or feature requests, please open an issue or submit a PR.
