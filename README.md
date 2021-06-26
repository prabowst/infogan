## InfoGAN - Study on MNIST Dataset

This project is inspired by the paper: [InfoGAN: Interpretable Representation Learning by Information Maximizing Generative Adversarial Nets](https://arxiv.org/abs/1606.03657). InfoGAN, or Information Maximizing GAN is an extension to GAN that has the ability to learn representations in an unsupervised manner. The project is following closely to the architecture of that presented in the paper with minor differences. The result is presented in various images and a gif highlighting the success of the model training process.

### Table of Contents 

1. [Requirements](#requirements)
2. [Project Description](#description)
3. [Files](#files)
4. [Project Results](#results)
5. [References](#references)

### Requirements<a name="requirements"></a>

The code runs using Python version 3. Below are the list of packages used within the scope of this project.

1. pandas
2. numpy 
3. time
4. tensorflow
5. tensorflow_probability
6. matplotlib
7. seaborn

### Project Description<a name="description"></a>

The project aims to reconstruct the Information Maximizing Generative Adversarial Network (InfoGAN) to learn the network capabilities in disentangle complex representations. The networks consist of a generator, discriminator and auxiliary. The generator produces fake image given probability distribution as noise, two continuous latent code and a single categorical code. There are 74 latent codes in total, 62 from noise, 2 from continuous code and 10 from categorical code (since the image classification relates to digits from 0 to 9).

The discriminator selectively chooses images that are real, from training dataset, and fake, from generator. The two networks play a minimax game in which that the generator is trying to fool the discriminator to produce fake images to be labelled real. An additional network is introduced, an auxiliary model, that can provide mutual information loss (minimizing the probability density of continuous codes and binary crossentropy of categorical code). This model is trained and share its loss values with both the generator and discriminator. It is hoped that this network architecture can generate valid images while obtaining specific meaning from the latent codes. 

### Files<a name="files"></a>

The files are organized as follows. 

```
- figures
|- image_generated_epoch_1.png
|- image_generated_epoch_10.png
|- image_generated_epoch_20.png
|- image_generated_epoch_30.png
|- image_generated_epoch_40.png
|- image_generated_epoch_50.png
|- network_losses.png

- gif
|- train_result.gif

- model
|- infogan_model_generator.tf

- generate_gif.py
- generate_sample_latent.py
- infogan.py
- model.py
- utils.py
- README.md
```

### Project Results<a name="results"></a>

The result of the training can be shown in several ways. The first aspect is the network losses. Both the generator and auxiliary's losses are decreasing in a stable fashion. This is ideal since it is preferred to have lower loss function for the generator to produce 'real' image and for the auxiliary model to produce less mutual info loss. Yet we can see that the discriminator's loss is increasing. Since the discriminator and the generator is playing a minmax game, it is most likely that the decreasing generator loss affects the increasing discriminator loss. 

![net_loss](https://github.com/prabowst/infogan/blob/master/figures/network_losses.png)

The following gif showcases the number of epochs and the quality of images generated by the generator model. 

![gif_train](https://github.com/prabbowst/infogan/blob/master/gif/train_result.gif)

Now let's take a look at the plots for varying c1 and c2.

![c1](https://github.com/prabowst/infogan/blob/master/figures/generated_vary_c1.png)
![c2](https://github.com/prabowst/infogan/blob/master/figures/generated_vary_c2.png)

From the 2 plots above, the categorical code is varied across the column. This can be considered a success that we see different digit in every row. Although it is obvious that the categorical code is learned unsupervised by the model, therefore code 1 does not necessarily correlate to number 1. Also another note, this categorical code isn't perfect in a sense that some of the digits are misclassified. 

As for the continuous code, c2 seems to control the thickness of the digit although it is somewhat responsible in the rotation of the image. But this is not as obvious as we can see from c1. Therefore, this can be a point of improvement as to create a network / preprocess the data prior to training so that the c1 and c2 can learn the latent representation of the images better. 

### References<a name="references"></a>

1. [InfoGAN: Interpretable Representation Learning by Information Maximizing Generative Adversarial Nets](https://arxiv.org/abs/1606.03657) 
2. https://github.com/daQuincy/InfoGAN_Tensorflow2.0