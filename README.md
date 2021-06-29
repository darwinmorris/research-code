# Conditional Comic Synthesis via Generative Adversarial Networks
This repository contains the code necessary to reproduce the results found in the paper "Condiitonal Comic Syntehsis via Generative Adversarial Networks". All implementations are in PyTorch. 

- Multi Label Auxiliary Stability GAN: The implementation for ML-SGAN can be found at the following repository: (https://github.com/darwinmorris/GAN_stability/blob/master/README.md)
- Multi Class Stability GAN: 
- DCGAN: Implimentation of DCGAN as found in the paper [Unsupervised Respresentation Learning With Deep Convolutional Generative Adversarial Networks](https://arxiv.org/pdf/1511.06434.pdf).
- WGAN-GP: 128 x 128 implementation of WGAN-GP. Follows the architecture found in [Improved Training of Wasserstein GANs](https://arxiv.org/pdf/1704.00028.pdf). Influenced by GAN implementations found [here](https://github.com/aladdinpersson/Machine-Learning-Collection).
- SGAN: The Stability GAN implementation follows the architecture displayed in the paper "[Which Training Methods for GANs do actually Converge?](https://avg.is.tuebingen.mpg.de/publications/meschedericml2018)". The paper implementation can be found [here](https://github.com/LMescheder/GAN_stability). Configs for experiments can be modified from the ML-SGAN repository.
- Data-prep/processing: Data preparation scripts for identifying and labeling background color, annotating comics, performing LP transformation, and calculating FID.
