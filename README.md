# variational-ae-tensorflow
This is a rather simple implementation of a variational autoencoder and conditional variational autoencoder




Credit to Carl Doersch https://arxiv.org/abs/1606.05908 

And whoever he gives credit to for coming up with the idea of VAEs. 


So this implementation follows the structure of the one from Doersch's paper (as well as his Caffe implementation which I looked at for reference).

HOWEVER the one big difference is that in the paper he mentions that with Z in less than 4 dimensions (and greater than 10000) he had trouble with getting it working properly. I too had that problem but I also wanted that pretty collage that you get from sampling z's from a VAE, so I ran it whilst printing out a lot of info including Log likelihood loss between X and X_hat, KL-div, the mean and std-dev of both Mu and Sigma that the encoder was generating and was noticing that what was messing it up was that when Z was 2d, my generated Sigma tended towards 0, and at one point it would hit it and completely screw up the backpropogation.

To fix it what I added was another component to the cost function which I called sigmaloss. 

    sigmaloss = (1-sigma)**4

Which I added to the previous two. What this does is more aggressively than KL-div push sigma towards 1, especially when it gets close to 0 or 2, whilst being very neutral when it's around 0.5~1.5 letting the backpropogation rely mostly on KL-div. This prevented sigma from actually reaching zero (though it still tends towards 0) which allowed for the model to train proper Gaussian distributions Q(Z|X) ~ P(Z).

This addition however leads to more variance among the generated Mu's, but in my experience that hasn't been that much of a problem looking at the last generated image you can see nice clusters of z coordinates that are clearly associated with certain number representations. However I do see multiple clusters for one or two of the numbers (e.g. 4 and 9 in the image in this repo) and this is probably due to that high variance of Mu's generated by the encoder. 





