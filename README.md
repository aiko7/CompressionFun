Project to compress down images per seperate class using a vgg/autoencoder architecture down into a small latent space and then reconstruct them from that, a different form of compression kind of.
Example of a 96% compression factor down into a latent space and then the following reconstruction achieved from some images from the tiny-imagenet-200 dataset.
<img width="1200" height="300" alt="image" src="https://github.com/user-attachments/assets/0225a482-8d00-4847-a558-e8fcb7b6c58b" />

Also included is a more lightweight version of the architecture which achieves a lesser compression factor but is set up to be nicely parallelizable per class on Azure ML.
At the current state the parralel version isn't fully tuned to achieve the best compression possible but was rather set up in a way for cost effective training and reasonable results.
Future updates will focused on greatly incression the parallel version compression factor without making model size and training costs explode.
