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


# select real samples
def generate_real_samples(images, n_samples):
    # choose random instances
    ix = randint(0, images.shape[0], n_samples)
    # select images
    X = images[ix]
    # generate class labels
    y = ones((n_samples, 1))
    return X, y


# generate points in latent space as input for the generator
def generate_latent_points(latent_dim, n_samples):
    # generate points in the latent space
    x_input = randn(latent_dim * n_samples)
    # reshape into a batch of inputs for the network
    x_input = x_input.reshape(n_samples, latent_dim)
    return x_input


# use the generator to generate n fake examples, with class labels
def generate_fake_samples(generator, latent_dim, n_samples):
    # generate points in latent space
    x_input = generate_latent_points(latent_dim, n_samples)
    # predict outputs
    X = generator.predict(x_input)
    # create class labels
    y = -ones((n_samples, 1))
    return X, y


# train a generator and discriminator
def epochs(g_model, d_model, gan_model, dataset, epochs, batch_size, latent_dim, fadein=False):
    # calculate the number of batches per training epoch
    # bat_per_epo = int(dataset.shape[0] / batch_size)
    # calculate the number of training iterations
    # n_steps = bat_per_epo * n_epochs
    # calculate the size of half a batch of samples
    # half_batch = int(n_batch / 2)
    # manually enumerate epochs
    print('* Batch size:', batch_size)
    print('* Iteration:', epochs)

    real = np.ones((batch_size, 1))
    fake = np.zeros((batch_size, 1))

    # pbar = trange(epochs, desc='Epoch runs', leave=True)
    # for i in pbar:
    for i in range(epochs):
        # update alpha for all WeightedSum layers when fading in new blocks
        # if fadein:
        #    update_fadein([g_model, d_model, gan_model], i, n_steps)

        idx = np.random.randint(0, dataset.shape[0], batch_size)
        imgs = dataset[idx]

        # prepare real and fake samples
        # X_real, y_real = generate_real_samples(dataset, half_batch)
        # X_fake, y_fake = generate_fake_samples(g_model, latent_dim, batch_size)
        z = np.random.normal(0, 1, (batch_size, latent_dim))
        gen_imgs = g_model.predict(z)

        # update discriminator model
        d_loss1 = d_model.train_on_batch(imgs, real)
        d_loss2 = d_model.train_on_batch(gen_imgs, fake)

        # update the generator via the discriminator's error
        # z_input = generate_latent_points(latent_dim, batch_size)
        z = np.random.normal(0, 1, (batch_size, latent_dim))
        gen_imgs = g_model.predict(z)
        # y_real2 = ones((n_batch, 1))

        g_loss = gan_model.train_on_batch(z, real)

        # summarize loss on this batch
        # pbar.set_postfix({'d1': "%.3f" % d_loss1, 'd2': "%.3f" % d_loss2, 'g': "%.3f" % g_loss})
        if (i % 200) == 0:
            print('>%d, d1=%.3f, d2=%.3f g=%.3f' % (i + 1, d_loss1, d_loss2, g_loss))
            generate(g_model, latent_dim)
            display.clear_output(wait=True)


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
