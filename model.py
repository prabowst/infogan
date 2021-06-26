import tensorflow as tf
import tensorflow.keras as keras 
import tensorflow_probability as tfp



def generator(input_size=74):

	'''
	Build the generator network: generate image to trick the discriminator.

	Params:
		input_size (int): the total size of latent space
	
	Return:
		generator_model (keras Model): generator network
	'''

	noise_input = keras.layers.Input(shape=(input_size,))
	dl_1 = keras.layers.Dense(1024, use_bias=False)(noise_input)
	bn_1 = keras.layers.BatchNormalization()(dl_1)
	af_1 = keras.layers.ReLU()(bn_1)

	dl_2 = keras.layers.Dense(128 * 7 * 7, use_bias=False)(af_1)
	bn_2 = keras.layers.BatchNormalization()(dl_2)
	af_2 = keras.layers.ReLU()(bn_2)

	reshape = keras.layers.Reshape((7, 7, 128))(af_2)

	ct_3 = keras.layers.Conv2DTranspose(64, (4,4), (2,2), padding='same', 
										use_bias=False)(reshape)
	bn_3 = keras.layers.BatchNormalization()(ct_3)
	af_3 = keras.layers.ReLU()(bn_3)

	output = keras.layers.Conv2DTranspose(1, (4,4), (2,2), padding='same', 
										  activation='tanh')(af_3)

	generator_model = keras.models.Model(noise_input, output)

	return generator_model

def discriminator(num_class=10, input_shape=(28,28,1)):

	'''
	Build the discriminator network to distinguish fake and real images;
	Construct the auxiliary model to provide information of 
	mutual info loss.

	Params:
		input_shape (tuple): the size of the training image
	
	Return:
		discriminator_model (keras Model): discriminator network
		auxiliary_model (keras Model): auxiliary network
	'''

	img_input = keras.layers.Input(shape=input_shape)

	cl_1 = keras.layers.Conv2D(64, (4,4), (2,2), padding='same', 
								use_bias=True)(img_input)
	af_1 = keras.layers.LeakyReLU(0.1)(cl_1)

	cl_2 = keras.layers.Conv2D(128, (4,4), (2,2), padding='same', 
								use_bias=False)(af_1)
	bn_2 = keras.layers.BatchNormalization()(cl_2)
	af_2 = keras.layers.LeakyReLU(0.1)(bn_2)
	fl_2 = keras.layers.Flatten()(af_2)

	dl_3 = keras.layers.Dense(1024, use_bias=False)(fl_2)
	bn_3 = keras.layers.BatchNormalization()(dl_3)
	af_3 = keras.layers.LeakyReLU(0.1)(bn_3)

	discriminator_output = keras.layers.Dense(1, activation='sigmoid')(af_3)

	qd_4 = keras.layers.Dense(128, use_bias=False)(af_3)
	bn_4 = keras.layers.BatchNormalization()(qd_4)
	af_4 = keras.layers.LeakyReLU(0.1)(bn_4)

	classification_output = keras.layers.Dense(num_class, 
							activation='softmax')(af_4)
	gaussian_mean_output = keras.layers.Dense(1)(af_4) # mu
	gaussian_stdev_output = keras.layers.Dense(1, 
							activation=lambda x:tf.math.exp(x))(af_4) # sigma

	discriminator_model = keras.models.Model(img_input, discriminator_output)
	auxiliary_model = keras.models.Model(img_input, 
		[classification_output, gaussian_mean_output, gaussian_stdev_output])

	return discriminator_model, auxiliary_model