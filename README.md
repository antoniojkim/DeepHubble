# DeepHubble

Explore GANs by building one that can generate high quality image of deep space (akin to images taken by the Hubble Telescope).

## Data

The data consists of images sampled from the larger Hubble eXtreme Deep Field (XDF).

![](https://cdn.spacetelescope.org/archives/images/screen/heic1214a.jpg)

## Experimentation

### [GAN](https://github.com/antoniojkim/DeepHubble/tree/master/model/DCGAN.ipynb)

Initial experimentation with a vanilla GAN as described in [this paper](https://arxiv.org/pdf/1406.2661.pdf) yielded unstable results.

The loss of the discriminator would tend towards zero while the loss of the generator would continue to increase. This tells us that the generator is performing very poorly and the discriminator is catching on and is able to easily identify which images are being generated and which come from the XDF image.
