from __future__ import print_function, division

import os

import pandas as pd
from keras.datasets import mnist
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, multiply
from keras.layers import BatchNormalization, Activation, Embedding, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras import metrics
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
from keras.optimizers import Adam
from keras.utils import to_categorical
from keras.layers.merge import concatenate

from load import load_wind
import plots
import scipy.misc


import numpy as np

class OHE_CGAN():
    def __init__(self, title):
        # Input shape
        self.img_rows = 1
        self.img_cols = 24
        self.num_classes = 4
        self.num_channels = 1
        self.img_shape = (self.img_rows, self.img_cols, self.num_channels)
        self.latent_dim = 100
        self.mean = 0.5
        self.std = 0.2
        self.g_losses = []
        self.d_losses = []
        self.title = title
        self.generator_in_channels = self.latent_dim + self.num_classes

        optimizer = Adam(0.0002, 0.5)

        # Build and compile the discriminator
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss=metrics.mean_squared_error,
            optimizer=optimizer,
            metrics=metrics.binary_accuracy)

        # Build the generator
        self.generator = self.build_generator()

        # The generator takes noise and the target label as input
        # and generates the corresponding digit of that label
        noise = Input(shape=(self.latent_dim,))
        label = Input(shape=(self.num_classes,))
        img = self.generator([noise, label])

        # For the combined model we will only train the generator
        self.discriminator.trainable = False

        # The discriminator takes generated image as input and determines validity
        # and the label of that image
        valid = self.discriminator([img, label])

        # The combined model  (stacked generator and discriminator)
        # Trains generator to fool discriminator
        self.combined = Model([noise, label], valid)
        self.combined.compile(loss=metrics.mean_squared_error,
            optimizer=optimizer)

    def build_generator(self):
        model = Sequential()
        # n_nodes = self.img_rows * self.img_cols * self.latent_dim
        model.add(Dense(256, input_dim=self.latent_dim+self.num_classes))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(512))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(1024))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(np.prod(self.img_shape,), activation='tanh'))
        model.add(Reshape(self.img_shape))

        noise = Input(shape=(self.latent_dim,))
        label = Input(shape=(self.num_classes,))
        #label_embedding = Flatten()(Embedding(self.num_classes, self.latent_dim)(label))
        model_input = concatenate([noise, label], axis=1)
        # model_input = multiply([noise, label_embedding])
        img = model(model_input)

        #rint('label embedding ', label_embedding.shape, label_embedding)
        print('g noise ', noise.shape)
        print('g label ', label.shape)
        print('g model input ', model_input.shape)
        print('g model ', img.shape)

        generator = Model([noise, label], img, name='generator')
        generator.summary()
        return generator

    def build_discriminator(self):
        img = Input(shape=self.img_shape,)
        label = Input(shape=(self.num_classes,), dtype='int32')
        x = img
        y = Dense(np.prod(self.img_shape))(label)
        y = Reshape(self.img_shape)(y)
        print('x = img ', x.shape)
        print('label ', label.shape)
        print('y ', y.shape)
        x = concatenate([x, y])

        x = LeakyReLU(alpha=0.2)(x)
        x = Dense(512)(x)
        x = LeakyReLU(alpha=0.2)(x)
        x = Dropout(0.4)(x)
        x = Dense(512)(x)
        x = LeakyReLU(alpha=0.2)(x)
        x = Dropout(0.4)(x)
        x = Flatten()(x)
        x = Dense(1, activation='sigmoid')(x)

        # model = Sequential()
        # model.add(Dense(512, input_dim=np.prod(self.img_shape))(label))
        # model.add(Reshape(self.img_shape)(label))
        # model.add(concatenate(img, label))
        # model.add(LeakyReLU(alpha=0.2))
        # model.add(Dense(512))
        # model.add(LeakyReLU(alpha=0.2))
        # model.add(Dropout(0.4))
        # model.add(Dense(512))
        # model.add(LeakyReLU(alpha=0.2))
        # model.add(Dropout(0.4))
        # model.add(Dense(1, activation='sigmoid'))


        print('x after concat ', x.shape)
        #label_embedding = Flatten()(Embedding(self.num_classes, np.prod(self.img_shape))(label))
        # flat_img = Flatten()(img)
        #model_input = multiply([flat_img, label_embedding])
        # model_input = concatenate([img, label], axis=1)
        # validity = model(model_input)
        print('d img ',img.shape)
        print('d label ',label.shape)
        print('d x ',x.shape)
        # print('d flat img ', flat_img.shape)
        # print('d model input ', model_input.shape)
        # print('d validity ', validity.shape)

        discriminator = Model([img, label], x, name='discriminator')
        discriminator.summary()
        return discriminator

    def train(self, epochs, batch_size=128, sample_interval=50):

        # Load the dataset
        # (X_train, y_train), (_, _) = mnist.load_data()
        X_train, y_train = load_wind()
        print('X TRAIN')
        print(X_train.shape)
        print('Y TRAIN')
        print(y_train.shape)

        # Configure input
        # X_train = (X_train.astype(np.float32) - 127.5) / 127.5
        X_train = np.expand_dims(X_train, axis=1)
        y_train = y_train.reshape(-1, 1)

        # Adversarial ground truths
        valid = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))

        # print('before OHE ', y_train.shape, y_train)
        ohe_labels = to_categorical(y_train, self.num_classes)
        # # ohe_labels = ohe_labels[:, :, None, None]
        # # ohe_labels = ohe_labels.reshape(y_train.shape[0], 4)
        # print('after OHE ', ohe_labels.shape, ohe_labels)
        for epoch in range(epochs):
            # ---------------------
            #  Train Discriminator
            # ---------------------
            # Select a random half batch of images
            idx = np.random.randint(0, X_train.shape[0], batch_size)

            imgs, labels = X_train[idx], ohe_labels[idx]

            # print("idx shape: ", idx.shape, idx)
            # print("imgs shape: ", imgs.shape)
            # print("labels shape: ", labels.shape)
            # Sample noise as generator input
            noise = np.random.normal(self.mean, self.std, (batch_size, self.latent_dim))

            # assign random one-hot labels
            fake_labels = np.eye(self.num_classes)[np.random.choice(self.num_classes, batch_size)]
            # Generate a half batch of new images
            gen_imgs = self.generator.predict([noise, labels])
            # print("gen imgs ", gen_imgs.shape)
            # print('gen labels ', fake_labels.shape)
            # print("gen_imgs imgs: ",gen_imgs)
            imgs = np.expand_dims(imgs, axis=3)
            # print('final imgs ', imgs.shape)
            # print('final labels ', labels.shape)
            # print('final valid ', valid.shape)
            # Train the discriminator
            d_loss_real = self.discriminator.train_on_batch([imgs, labels], valid)
            d_loss_fake = self.discriminator.train_on_batch([gen_imgs, labels], fake)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            # ---------------------
            #  Train Generator
            # ---------------------
            # Condition on labels
            #sampled_labels = np.random.randint(0, 4, batch_size).reshape(-1, 1)
            fake_labels = np.eye(self.num_classes)[np.random.choice(self.num_classes, batch_size)]
            #print('fake labels ', fake_labels.shape)
            # Train the generator
            g_loss = self.combined.train_on_batch([noise, fake_labels], valid)

            # Plot the progress
            print ("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100*d_loss[1], g_loss))
            self.g_losses.append(g_loss)
            self.d_losses.append(d_loss[0])
            # self.w_losses.append(w_loss)

            # If at save interval => save generated image samples
            if epoch % sample_interval == 0:
                save = False
                script_dir = os.path.dirname(__file__)
                results_dir = os.path.join(script_dir, 'images/OHE CGAN/%s %d/' %(self.title, epoch))
                if epoch == epochs - 1:
                    save = True
                plots.sample_plots(self, save, results_dir)
                plots.plot_loss(self, save, results_dir)
                plots.distributions(self, save, imgs, gen_imgs, results_dir)

    # Defining Wasserstein Loss
    def wasserstein_loss(self, y_true, y_pred):
        return np.mean(y_true * y_pred)

    def OneHot(self, X, n, negative_class=0.):
        X = np.asarray(X).flatten()
        if n is None:
            n = np.max(X) + 1
        Xoh = np.ones((len(X), n)) * negative_class
        for i in range(len(X)):
            m=X[i]
            Xoh[i,m]=1
        return Xoh

    # def sample_images(self, epoch):
    #     r, c = 2, 5
    #     noise = np.random.normal(0, 1, (r * c, 100))
    #     sampled_labels = np.arange(0, 10).reshape(-1, 1)
    #
    #     gen_imgs = self.generator.predict([noise, sampled_labels])
    #
    #     # Rescale images 0 - 1
    #     gen_imgs = 0.5 * gen_imgs + 0.5
    #
    #     fig, axs = plt.subplots(r, c)
    #     cnt = 0
    #     for i in range(r):
    #         for j in range(c):
    #             axs[i,j].imshow(gen_imgs[cnt,:,:,0], cmap='gray')
    #             axs[i,j].set_title("Digit: %d" % sampled_labels[cnt])
    #             axs[i,j].axis('off')
    #             cnt += 1
    #     fig.savefig("images/%d.png" % epoch)
    #     plt.close()



if __name__ == '__main__':
    title = 'MSExMSE'
    epochs = 10000

    sample_interval = (epochs) / 10
    cgan = OHE_CGAN(title)
    cgan.train(epochs=epochs+1, batch_size=32, sample_interval=sample_interval)
