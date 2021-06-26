import tensorflow as tf
import tensorflow.keras as keras 
import tensorflow_probability as tfp

import matplotlib.pyplot as plt
import numpy as np



def sample_generator_input(batch_size, con_size, num_class):

	'''
	Generate sample input distribution for the generator. This input 
	will provide information for the fake image generation. In InfoGAN 
	the categorical and continuous codes are included for latent space learning.

	Params:
		batch_size (int): batch training size
		con_size (int): noise size
		num_class (int): number of categorical size in training image

	Return:
		z_cat (array): categorical code
		z_c01 (array): continuous code 1
		z_c02 (array): continuous code 2
		z_con (array): noise
	'''

	z_con = tf.random.normal([batch_size, con_size])

	z_cat = tf.random.uniform([batch_size], 0, 10, dtype=tf.int32)
	z_cat = tf.one_hot(z_cat, num_class)

	z_c01 = tf.random.uniform([batch_size, 1], -1, 1)
	z_c02 = tf.random.uniform([batch_size, 1], -1, 1)

	return z_cat, z_c01, z_c02, z_con

def plot_interval(epoch, gen_model):

	'''
	Generate sample plot based on the completed number of epochs.

	Params:
		epoch (int): current training loop
		gen_model (keras Model): trained generator network

	Return:
		None
	'''

	row, col = 5, 5
	fig, axes = plt.subplots(row, col)
	for i in range(col):
		label = tf.keras.utils.to_categorical(np.full(fill_value=i, 
				shape=(row,1)), num_classes=10)
		_, con_1, con_2, noise = sample_generator_input(col, 62, 10)
		gen_input = np.concatenate((label, con_1, con_2, noise), axis=1)
		gen_image = gen_model(gen_input, training=False)
		gen_image = (gen_image / 2) + 0.5
		for j in range(row):
			axes[j,i].imshow(gen_image[j,:,:,0], cmap='gray')
			axes[j,i].axis('off')
	plt.suptitle('Sample Images Epoch ' + str(epoch))
	plt.savefig('figures/image_generated_epoch_' + str(epoch) + '.png', 
				dpi=300, bbox_inches='tight')
	plt.close()