This repository contains scripts to process, cluster and analyze images of bacterial colonies.

Images are cropped to single colonies and their background is subtracted. Then features are extracted either manually or by neural network, 
such as different autoencoder models. The resulting latent space is then used to cluster the colonies by their morphology. An example is shown in the image below:

![alt text](https://github.com/tcapraz/biofilm_screen/blob/main/colony_clustering_example.png?raw=true)
