# UNITN-Project-Course2023-2024

# Brain Tumor Image Reconstruction using Autoencoders

## Overview
This project implements two types of autoencoder models (Simple Autoencoder and Variational Autoencoder) for reconstructing brain tumor images from the Brain Tumor MRI Images Dataset available on Hugging Face. The main goals are to understand the working of each model, compare their performance, visualize intermediate layers and reconstructed images, and explore the impact of hyperparameter tuning.

## Objectives
- Visualize the intermediate layers and explain their functionality.
- Compare the performance of Simple Autoencoder and Variational Autoencoder in terms of image reconstruction.
- Perform hyperparameter tuning to obtain the best model for image reconstruction.
- Evaluate the trained best model's suitability for image denoising.

## Dataset
- **Source**: Brain Tumor MRI Images Dataset from Hugging Face
- **Training Images**: 2870
- **Testing Images**: 394
- **Image Dimensions**: 512 x 512 pixels (resized to 128 x 128 pixels for this project)
- **Image Type**: Grayscale

## Methodology

### 1. Simple Autoencoder (AE)
- **Architecture**: 5 convolutional layers for the encoder and 5 deconvolutional layers for the decoder.
- **Activation Function**: Leaky ReLU
- **Loss Function**: Mean Squared Error (MSE)
- **Purpose**: Compress input images into a lower-dimensional representation and reconstruct them.

### 2. Variational Autoencoder (VAE)
- **Architecture**: Similar to AE with 5 convolutional layers for the encoder and 5 deconvolutional layers for the decoder.
- **Additional Features**: Uses probabilistic encoders and decoders, incorporates reparameterization trick.
- **Loss Function**: Combination of Binary Cross-Entropy (BCE) and Kullback-Leibler (KL) Divergence.
- **Purpose**: Learn the distribution of the data to generate and reconstruct new data similar to the input.

### Hyperparameter Optimization
- **Tool**: Weights and Biases (W&B) Sweeps
- **Optimization Method**: Bayesian optimization
- **Hyperparameters Tuned**:
  - Learning Rate: [1e-3, 5e-4, 1e-4]
  - Epochs: [10, 20, 40]
  - Batch Size: [16, 32, 64]
  - Latent Dimension: [512, 1024, 2048]

### Image Denoising
- **Approach**: Using the best-trained autoencoder model to denoise images with added Gaussian noise.

## Results and Discussion

### Image Reconstruction
- **Simple Autoencoder**: Achieved high fidelity in reconstructing images, particularly in areas with clear black and white differentiation.
- **Variational Autoencoder**: Reconstructed images were blurrier due to the stochastic nature of VAEs.

### Intermediate Layer Visualization
- Visualization of intermediate layers showed the gradual transition from capturing basic features to reconstructing detailed structures.

### Denoising
- The best AE model successfully removed strong noise while preserving important features, though some important features like brain tumors were occasionally lost.

## Future Work
- Explore data preprocessing techniques like normalization.
- Further optimize hyperparameters including the number of filters, kernel size, and stride.
- Investigate different models such as U-Net and Autoencoders with Skip Connections.

## Conclusion
- The Simple Autoencoder outperformed the Variational Autoencoder in image reconstruction.
- The best AE model showed potential for image denoising but needs improvement to avoid loss of important features.
- Visualization of intermediate layers provided insights into the model's processing and reconstruction capabilities.

## References
1. Zhai, J., Zhang, S., Chen, J., & He, Q. (2018). Autoencoder and Its Various Variants.
2. Vincent, P., et al. (2010). Stacked Denoising Autoencoders.
3. Sakurada, M., & Yairi, T. (2014). Anomaly Detection Using Autoencoders.
4. Brain Tumor MRI Images Dataset: [Hugging Face](https://huggingface.co/datasets/benschill/brain-tumor-collection)
5. Kingma, D. P., & Welling, M. (2014). Auto-Encoding Variational Bayes.
6. Weights and Biases: [W&B](https://wandb.ai/site)
