# Dependencies
from mnist import MNIST
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from numpy import linalg as LA

filter_size = 10 # how many kernels wide and tall the filter is
kernel_size = 7 # how many pixels wide and tall the kernel is
image_size = 28 # pixels
stride = kernel_size # possibly play with overlapping convolution

mndata = MNIST('./')
images, labels = mndata.load_training()

# make matplotlib images
def plot_image(image, name='plot.png', dim=2):
	if dim==4:
		pixels = filter_size * kernel_size
		image = image.swapaxes(1,2).reshape(pixels,pixels)
	plt.figure(2)
	plt.imshow(image, cmap='Greys_r', interpolation='none')
	plt.savefig(name)

# train() takes a set of images, and two optional arguments:
# delta is the rate of change for the nearest neighbor and 
# buckshot is the rate of change for that neighbor's neighbors
def train(image_set, delta=.5, buckshot=.1):
	filt = np.random.randint(2, size=(filter_size, filter_size, kernel_size, kernel_size))
	for image in image_set:
		image = np.array(image).reshape(image_size,image_size)
		for a in range(0,len(image),stride):
			for b in range(0, len(image), stride):
				target = image[a : a + stride, b : b + stride]
				flat_filt = filt.reshape(filter_size**2, kernel_size**2)
				nn_list = list(map(lambda x: np.linalg.norm(target.flatten() - x), flat_filt))
				nn_index = np.argmin(nn_list)
				x,y = nn_index / filter_size, nn_index % filter_size
				for i in [-1,0,1]:
					for j in [-1,0,1]:
						if 0 <= x+i < filter_size and 0 <= y+j < filter_size:
							d = buckshot if i**2+j**2 > 0 else delta
							filt[x+i][y+j] = filt[x+i][y+j] + d*(target - filt[x+i][y+j])
	return filt

# Execution. Train the filter on the first 500 images, then create file with updated filter.
filt = train(images[:500])

print 'image shape', (image_size, image_size)
print 'filter shape ', filt.shape

plot_image(filt, 'filter.png', dim=4)
plot_image(np.array(images[1]).reshape(image_size, image_size), 'image.png')

