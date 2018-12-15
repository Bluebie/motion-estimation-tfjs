This is a little thingie that uses some sample photos to train a tiny network to do simple motion estimation with tensorflow

Images must be supplied in sample-pics, and must all have the exact same width and height and number of channels, but can be an arbitrary shape, though should be much larger than the patch size that the network will be learning.

Images are cropped in to little offset squares, and then the network is taught to estimate the amount of motion in x and y using the known random offset