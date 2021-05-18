from tqdm import trange
import numpy as np
from numpy.random import randint
from numpy.random import randn
from numpy import ones
from math import sqrt
import matplotlib.pyplot as plt
import tensorflow as tf
import gan
from tensorflow.keras import backend

from IPython import display


generator_optimizer = tf.keras.optimizers.Adam(1e-4)
discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)


def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss


def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)


# @tf.function
def train_step(discriminator, generator, images, batch_size, latent_dim):
    noise = tf.random.normal([batch_size, latent_dim])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = generator(noise, training=True)

        real_output = discriminator(images, training=True)
        fake_output = discriminator(generated_images, training=True)

        gen_loss = generator_loss(fake_output)
        disc_loss = discriminator_loss(real_output, fake_output)

    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

    return disc_loss, gen_loss


# generate samples and save as a plot and save the model
def generate(generator, latent_dim, n_samples=16, clear=True):
    if clear:
        display.clear_output(wait=True)

    # generate images
    z = np.random.normal(0, 1, (n_samples, latent_dim))
    X = generator.predict(z)

    X = (X - X.min()) / (X.max() - X.min())
    # plot real images
    square = int(sqrt(n_samples))
    for i in range(n_samples):
        plt.subplot(square, square, 1 + i)
        plt.axis('off')
        plt.imshow(X[i])
    plt.show()


def update_fadein(models, epoch, epochs):
    # calculate current alpha (linear from 0 to 1)
    alpha = epoch / float(epochs - 1)
    # update the alpha for each model
    for model in models:
        for layer in model.layers:
            if isinstance(layer, gan.WeightedSum):
                backend.set_value(layer.alpha, alpha)
