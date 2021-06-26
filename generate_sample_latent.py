import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

gen_model = tf.keras.models.load_model('model/infogan_model_generator.tf', compile=False)

row, col = 10, 10
batch_size = row * col
con_size = 62

# vary z_c01

image_list = []
for i in range(col):
    image_row = []
    z_cat = tf.keras.utils.to_categorical(np.full(fill_value=i, 
				shape=(row,1)), num_classes=10)
    z_c01 = tf.convert_to_tensor(np.linspace(-1.5, 1.5, col).reshape(col, 1))
    z_c02 = tf.convert_to_tensor(np.zeros((col, 1)))
    z_con = tf.random.normal([col, con_size])
    gen_input = np.concatenate((z_cat, z_c01, z_c02, z_con), axis=1)
    gen_image = gen_model(gen_input, training=False)
    for j in range(row):
        image = gen_image[j,:,:,0]
        image_row.append(image)
    image_row = np.concatenate(image_row, 1)
    image_list.append(image_row)

image_list = np.concatenate(image_list, 0)

plt.figure(figsize=(16,16))
plt.title('Generated Image with Varying c1 [-1.5 to 1.5]')
plt.imshow(image_list, cmap='gray')
plt.axis('off')
plt.savefig('figures/generated_vary_c1.png', dpi=300, bbox_inches='tight')
plt.close()

# vary z_c02

image_list = []
for i in range(col):
    image_row = []
    z_cat = tf.keras.utils.to_categorical(np.full(fill_value=i, 
				shape=(row,1)), num_classes=10)
    z_c02 = tf.convert_to_tensor(np.linspace(-1.5, 1.5, col).reshape(col, 1))
    z_c01 = tf.convert_to_tensor(np.zeros((col, 1)))
    z_con = tf.random.normal([col, con_size])
    gen_input = np.concatenate((z_cat, z_c01, z_c02, z_con), axis=1)
    gen_image = gen_model(gen_input, training=False)
    for j in range(row):
        image = gen_image[j,:,:,0]
        image_row.append(image)
    image_row = np.concatenate(image_row, 1)
    image_list.append(image_row)

image_list = np.concatenate(image_list, 0)

plt.figure(figsize=(16,16))
plt.title('Generated Image with Varying c2 [-1.5 to 1.5]')
plt.imshow(image_list, cmap='gray')
plt.axis('off')
plt.savefig('figures/generated_vary_c2.png', dpi=300, bbox_inches='tight')
plt.close()