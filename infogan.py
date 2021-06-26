import tensorflow as tf
import tensorflow.keras as keras 
import tensorflow_probability as tfp

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import time

from model import generator, discriminator
from utils import sample_generator_input, plot_interval

import warnings
warnings.filterwarnings('ignore')

def train():

	'''
	Training loop of InfoGAN. This includes the declaration of the networks, specifications of optimizers, learning rate, batch size, and gradient calculations.

	Params:
		None

	Return:
		None
	'''

	(X_img, y_img), (_, _) = tf.keras.datasets.mnist.load_data()

	X_img = X_img.reshape((X_img.shape[0], 28, 28, 1))
	X_img = (X_img/127.5) - 1

	gen_optim = keras.optimizers.Adam(1e-3)
	disc_optim = keras.optimizers.Adam(2e-4)
	aux_optim = keras.optimizers.Adam(2e-4)

	gen_model = generator()
	disc_model, aux_model = discriminator()

	batch = 128
	con_size = 62
	num_class = 10
	epochs = 50

	disc_losses = []
	gen_losses = []
	aux_losses = []

	for epoch in range(epochs):

		temp_disc, temp_gen, temp_aux = [], [], []

		start = time.time()
		X_dataset = tf.data.Dataset.from_tensor_slices(X_img) \
					.shuffle(X_img.shape[0]).batch(batch)
		num_step = 0

		for X_batch in X_dataset:

			'==================TRAIN_STEP==================='

			losses = [
				keras.losses.BinaryCrossentropy(),
				keras.losses.CategoricalCrossentropy()
			]

			batch_size = X_batch.shape[0]

			gen_cat, gen_c1, gen_c2, gen_con = sample_generator_input(batch_size, con_size, num_class) 
			gen_input = np.concatenate((gen_cat, gen_c1, gen_c2, gen_con), axis=1)

			with tf.GradientTape() as discriminator_tape:
				disc_model.trainable = True
				discriminator_tape.watch(disc_model.trainable_variables)

				disc_real_out = disc_model(X_batch, training=True)
				disc_real_loss = losses[0](tf.ones((batch_size, 1)), disc_real_out)

				image_fake = gen_model(gen_input, training=True)
				disc_fake_out = disc_model(image_fake, training=True)
				disc_fake_loss = losses[0](tf.zeros((batch_size, 1)), disc_fake_out)

				disc_loss = disc_real_loss + disc_fake_loss

			disc_grad = discriminator_tape.gradient(disc_loss, disc_model.trainable_variables)
			disc_optim.apply_gradients(zip(disc_grad, disc_model.trainable_variables))

			batch_size = batch_size * 2

			with tf.GradientTape() as generator_tape, tf.GradientTape() as aux_tape:
				generator_tape.watch(gen_model.trainable_variables)
				aux_tape.watch(aux_model.trainable_variables)

				gen_cat, gen_c1, gen_c2, gen_con = sample_generator_input(batch_size, con_size, num_class)
				gen_input = np.concatenate((gen_cat, gen_c1, gen_c2, gen_con), axis=1)

				image_fake = gen_model(gen_input, training=True)
				disc_fake_out = disc_model(image_fake, training=True)
				gen_image_loss = losses[0](tf.ones(batch_size, 1), disc_fake_out)

				cat, mu, sigma = aux_model(image_fake, training=True)
				cat_loss = losses[1](gen_cat, cat)

				gauss_dist = tfp.distributions.Normal(mu, sigma)

				c1_loss = tf.reduce_mean(-gauss_dist.log_prob(gen_c1))
				c2_loss = tf.reduce_mean(-gauss_dist.log_prob(gen_c2))

				gen_loss = gen_image_loss + cat_loss + c1_loss + c2_loss
				aux_loss = cat_loss + c1_loss + c2_loss

			disc_model.trainable = False

			gen_grad = generator_tape.gradient(gen_loss, gen_model.trainable_variables)
			aux_grad = aux_tape.gradient(aux_loss, aux_model.trainable_variables)

			gen_optim.apply_gradients(zip(gen_grad, gen_model.trainable_variables))
			aux_optim.apply_gradients(zip(aux_grad, aux_model.trainable_variables))

			temp_disc.append(disc_loss)
			temp_gen.append(gen_loss)
			temp_aux.append(aux_loss)

			num_step += 1
			if num_step >= 100:
				break

		if ((epoch+1) % 10 == 0) or (epoch == 0):
			plot_interval(epoch+1, gen_model)
			if (epoch+1) % 25 == 0:
				gen_model.save('model/infogan_model_generator.tf')

		disc_losses.append(np.mean(temp_disc))
		gen_losses.append(np.mean(temp_gen))
		aux_losses.append(np.mean(temp_aux))

		print('Epoch [{:2d}/{:2d}] | disc_loss: {:6.4f} | gen_loss: {:6.4f} | aux_loss: {:6.4f} | runtime: {:.2f}s' \
		.format(epoch+1, epochs, np.mean(temp_disc), np.mean(temp_gen), np.mean(temp_aux), time.time()-start))

	epoch_axis = np.arange(1, (epochs)+1, dtype=np.int32)

	df = pd.DataFrame(index=epoch_axis)
	df['epoch'] = df.index
	df['disc_loss'] = disc_losses
	df['gen_loss'] = gen_losses
	df['aux_loss'] = aux_losses

	df = pd.melt(df, id_vars=['epoch'], value_vars=['disc_loss', 'gen_loss', 'aux_loss'],
					 var_name='loss_type', value_name='loss')

	sns.set_style('white')
	plt.figure(figsize=(8,6))
	ax = sns.lineplot(data=df, x='epoch', y='loss', hue='loss_type', marker='o')
	ax.set_title('Network Losses')
	plt.savefig('figures/network_losses.png', dpi=300, bbox_inches='tight')
	plt.close()

if __name__ == '__main__':
	train()