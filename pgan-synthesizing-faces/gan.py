import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import layers
from tensorflow.keras.layers import Add
from tensorflow.keras import backend


# weighted sum output
class WeightedSum(Add):
    # init with default value
    def __init__(self, alpha=0.0, **kwargs):
        super(WeightedSum, self).__init__(**kwargs)
        self.alpha = backend.variable(alpha, name='ws_alpha')

    # output a weighted sum of inputs
    def _merge_function(self, inputs):
        # only supports a weighted sum of two inputs
        assert (len(inputs) == 2)
        # ((1-a) * input1) + (a * input2)
        output = ((1.0 - self.alpha) * inputs[0]) + (self.alpha * inputs[1])
        return output


# Discriminator

# add a discriminator block
def add_discriminator_block(old_model, n_input_layers=3):
    print("Creating a new discriminator with improved shape")
    # get shape of existing model
    in_shape = list(old_model.input.shape)
    print("\t* Previous model: {}".format(old_model.input.shape[1:4]))

    # define new input shape as double the size
    input_shape = (in_shape[-2] * 2, in_shape[-2] * 2, in_shape[-1])
    in_image = layers.Input(shape=input_shape)
    print("\t* New model: {}".format(in_image.shape[1:4]))

    # Define new input processing layer
    d = layers.Conv2D(64, (1, 1), padding='same', kernel_initializer='he_normal')(in_image)
    d = layers.LeakyReLU(alpha=0.2)(d)

    # define new block
    d = layers.Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal')(d)
    d = layers.BatchNormalization()(d)
    d = layers.LeakyReLU(alpha=0.2)(d)
    d = layers.Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal')(d)
    d = layers.BatchNormalization()(d)
    d = layers.LeakyReLU(alpha=0.2)(d)
    d = layers.AveragePooling2D()(d)
    block_new = d

    # skip the input, 1x1 and activation for the old model
    for i in range(n_input_layers, len(old_model.layers)):
        d = old_model.layers[i](d)

    # define straight-through model
    model1 = keras.Model(in_image, d)
    # model1.compile(loss='mse', optimizer=Adam(lr=0.001, beta_1=0, beta_2=0.99, epsilon=10e-8))

    # down sample the new larger image
    downsample = layers.AveragePooling2D()(in_image)

    # connect old input processing to down sampled new input
    block_old = old_model.layers[1](downsample)
    block_old = old_model.layers[2](block_old)

    # fade in output of old model input layer with new input
    d = WeightedSum()([block_old, block_new])

    # skip the input, 1x1 and activation for the old model
    for i in range(n_input_layers, len(old_model.layers)):
        d = old_model.layers[i](d)

    # define straight-through model
    model2 = keras.Model(in_image, d)
    # model2.compile(loss='mse', optimizer=Adam(lr=0.001, beta_1=0, beta_2=0.99, epsilon=10e-8))

    return [model1, model2]


# define the discriminator models for each image resolution
def build_discriminator(input_shape=(4, 4, 3)):
    print('Building the Discriminator')
    print("\t* input shape: {}".format(input_shape))

    # base model input
    in_image = keras.Input(shape=input_shape)

    # conv 1x1
    d = layers.Conv2D(64, (1, 1), padding='same', kernel_initializer='he_normal')(in_image)
    d = layers.LeakyReLU(alpha=0.2)(d)

    # conv 3x3 (output block)
    d = layers.Conv2D(128, (3, 3), padding='same', kernel_initializer='he_normal')(d)
    d = layers.BatchNormalization()(d)
    d = layers.LeakyReLU(alpha=0.2)(d)

    # conv 4x4
    d = layers.Conv2D(128, (4, 4), padding='same', kernel_initializer='he_normal')(d)
    d = layers.BatchNormalization()(d)
    d = layers.LeakyReLU(alpha=0.2)(d)

    # dense output layer
    d = layers.Flatten()(d)
    out_class = layers.Dense(1)(d)

    # define model and compile
    model = keras.Model(in_image, out_class, name="Discriminator")
    # model.compile(loss='mse', optimizer=Adam(lr=0.001, beta_1=0, beta_2=0.99, epsilon=10e-8))

    print('\t* Discriminator compiled')
    return model


def discriminators(n_blocks, input_shape=(4, 4, 3)):
    model = build_discriminator(input_shape)

    # store model
    model_list = list()
    model_list.append([model, model])

    # create submodels
    print("Creating discriminator sub-models")
    for i in range(1, n_blocks):
        # get prior model without the fade-on
        old_model = model_list[i - 1][0]
        # create new model for next resolution
        models = add_discriminator_block(old_model)
        # store model
        model_list.append(models)
    print("{} discriminator models created".format(len(model_list)))
    return model_list


# Generator

# add a generator block
def add_generator_block(old_model):
    print("Creating a new generator with improved shape")
    # get the end of the last block
    block_end = old_model.layers[-2].output

    # upsample, and define new block
    upsampling = layers.UpSampling2D()(block_end)

    g = layers.Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal')(upsampling)
    g = layers.BatchNormalization()(g)
    g = layers.LeakyReLU(alpha=0.2)(g)
    g = layers.Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal')(g)
    g = layers.BatchNormalization()(g)
    g = layers.LeakyReLU(alpha=0.2)(g)
    # add new output layer
    out_image = layers.Conv2D(3, (1, 1), padding='same', kernel_initializer='he_normal')(g)
    # define model
    model1 = keras.models.Model(old_model.input, out_image)
    # get the output layer from old model
    out_old = old_model.layers[-1]
    # connect the upsampling to the old output layer
    out_image2 = out_old(upsampling)
    # define new output image as the weighted sum of the old and new models
    merged = WeightedSum()([out_image2, out_image])
    # define model
    model2 = keras.models.Model(old_model.input, merged)
    return [model1, model2]


# define generator models
def build_generator(latent_dim, in_dim=4):
    print('Building the Generator')
    # base model latent input
    in_latent = keras.Input(shape=(latent_dim,))

    # linear scale up to activation maps
    g = layers.Dense(128 * in_dim * in_dim, kernel_initializer='he_normal')(in_latent)
    g = layers.Reshape((in_dim, in_dim, 128))(g)

    # conv 4x4, input block
    g = layers.Conv2D(128, (3, 3), padding='same', kernel_initializer='he_normal')(g)
    g = layers.BatchNormalization()(g)
    g = layers.LeakyReLU(alpha=0.2)(g)

    # conv 3x3
    g = layers.Conv2D(128, (3, 3), padding='same', kernel_initializer='he_normal')(g)
    g = layers.BatchNormalization()(g)
    g = layers.LeakyReLU(alpha=0.2)(g)

    # conv 1x1, output block
    out_image = layers.Conv2D(3, (1, 1), padding='same', kernel_initializer='he_normal')(g)

    # define model
    return keras.models.Model(in_latent, out_image)


def generators(latent_dim, n_blocks, in_dim=4):
    model = build_generator(latent_dim, in_dim)

    # store model
    model_list = list()
    model_list.append([model, model])
    # create submodels
    for i in range(1, n_blocks):
        # get prior model without the fade-on
        old_model = model_list[i - 1][0]
        # create new model for next resolution
        models = add_generator_block(old_model)
        # store model
        model_list.append(models)
    print("{} generator models created".format(len(model_list)))
    return model_list
