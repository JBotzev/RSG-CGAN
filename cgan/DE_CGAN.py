from __future__ import print_function, division

import os

import pandas as pd
from keras.datasets import mnist
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, multiply
from keras.layers import BatchNormalization, Activation, Embedding, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D

from keras.models import Sequential, Model
from keras.optimizers import Adam
from keras import metrics

from load import load_wind
import plots
import wandb
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from wandb.keras import WandbCallback

import numpy as np

class DE_CGAN():
    def __init__(self):
        # Input shape
        self.img_rows = 1
        self.img_cols = 24
        self.num_classes = 4
        self.num_channels = 1
        self.img_shape = (self.img_rows, self.img_cols, self.num_channels)
        self.latent_dim = 100
        self.mean = 0.5
        self.std = 1
        self.learn_rate = 0.0002
        self.g_losses = []
        self.d_losses = []
        self.title = 'bi'
        self.generator_in_channels = self.latent_dim + self.num_classes

        optimizer = Adam(self.learn_rate, 0.5)

        # Build and compile the discriminator
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss=[metrics.binary_crossentropy],
                                   optimizer=optimizer,
                                   metrics=metrics.binary_accuracy)

        # Build the generator
        self.generator = self.build_generator()

        # The generator takes noise and the target label as input
        # and generates the corresponding digit of that label
        noise = Input(shape=(self.latent_dim,))
        label = Input(shape=(1,))
        img = self.generator([noise, label])

        # For the combined model we will only train the generator
        self.discriminator.trainable = False

        # The discriminator takes generated image as input and determines validity
        # and the label of that image
        valid = self.discriminator([img, label])

        # The combined model  (stacked generator and discriminator)
        # Trains generator to fool discriminator
        self.combined = Model([noise, label], valid)
        self.combined.compile(loss=metrics.binary_crossentropy,
                              optimizer=optimizer)

    def build_generator(self):
        model = Sequential()
        # n_nodes = self.img_rows * self.img_cols * self.latent_dim
        model.add(Dense(256, input_dim=self.latent_dim))
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
        label = Input(shape=(1,))
        label_embedding = Flatten()(Embedding(self.num_classes, self.latent_dim)(label))
        model_input = multiply([noise, label_embedding])
        validity = model(model_input)
        print('g noise ', noise.shape)
        print('g label ', label.shape)
        print('g label emb ', label_embedding.shape)
        print('g model_input ', model_input.shape)
        print('g validity ', validity.shape)

        generator = Model([noise, label], validity, name='generator')
        generator.summary()
        return generator

    def build_discriminator(self):
        model = Sequential()
        model.add(Dense(512, input_dim=np.prod(self.img_shape)))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(512))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.4))
        model.add(Dense(512))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.4))
        model.add(Dense(1, activation='sigmoid'))

        img = Input(shape=self.img_shape,)
        label = Input(shape=(1,), dtype='int32')
        label_embedding = Flatten()(Embedding(self.num_classes, np.prod(self.img_shape))(label))
        flat_img = Flatten()(img)
        model_input = multiply([flat_img, label_embedding])
        validity = model(model_input)
        print('d img ',img.shape)
        print('d label ',label.shape)
        print('d label emb',label_embedding.shape)
        print('d flat img ', flat_img.shape)
        print('d model input ', model_input.shape)
        print('d validity ', validity.shape)

        discriminator = Model([img, label], validity, name='discriminator')
        discriminator.summary()
        return discriminator

    def train(self, X_train, y_train, epochs, batch_size=128, sample_interval=50):

        # Load the dataset
        # (X_train, y_train), (_, _) = mnist.load_data()

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

        for epoch in range(epochs):
            # ---------------------
            #  Train Discriminator
            # ---------------------
            # Select a random half batch of images
            idx = np.random.randint(0, X_train.shape[0], batch_size)

            imgs, labels = X_train[idx], y_train[idx]

            # print("idx shape: ", idx.shape, idx)
            # print("imgs shape: ", imgs.shape)
            # print("labels shape: ", labels.shape)
            # Sample noise as generator input
            noise = np.random.normal(self.mean, self.std, (batch_size, self.latent_dim))

            # Generate a half batch of new images
            gen_imgs = self.generator.predict([noise, labels])
            # print("gen imgs ", gen_imgs.shape)

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
            sampled_labels = np.random.randint(0, 4, batch_size).reshape(-1, 1)
            # Train the generator
            g_loss = self.combined.train_on_batch([noise, sampled_labels], valid)

            # Plot the progress
            print ("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100*d_loss[1], g_loss))
            self.g_losses.append(g_loss)
            self.d_losses.append(d_loss[0])
            # self.w_losses.append(w_loss)

            # If at save interval => save generated image samples
            if epoch % sample_interval == 0:
                save = False
                script_dir = os.path.dirname(__file__)
                results_dir = os.path.join(script_dir, 'images/Decimal CGAN/%s/%d-(%0.1f,%0.1f)-noise%d-batch%d-lr%0.4f-[256,512,1024,512,512,512]/' %(self.title, epochs-1,self.mean, self.std,self.latent_dim,batch_size,self.learn_rate))
                if epoch == epochs - 1:
                    save = True

                # Plot and save statisticis
                plots.sample_plots(self, True, results_dir, epoch)
                plots.plot_loss(self, True, results_dir, epochs)
                #plots.distributions(self, True, imgs, gen_imgs, results_dir, epoch)
                plots.labeled_distributions(self, True, X_train, y_train, results_dir, epoch)

if __name__ == '__main__':
    # wandb.init(project="my-test-project", entity="joanbotzev")

    epochs = 10000
    batch_size = 64
    learning_rate = 0.0002
    # wandb.config = {
    #     "learning_rate": learning_rate,
    #     "epochs": epochs,
    #     "batch_size": batch_size
    # }

    sample_interval = (epochs) / 20
    X, y = load_wind()
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

    cgan = DE_CGAN()
    cgan.train(X, y, epochs=epochs+1, batch_size=batch_size, sample_interval=sample_interval)
    #cgan.combined.fit(X_train, y_train, validation_data=(X_test, y_test), callbacks=[WandbCallback()])