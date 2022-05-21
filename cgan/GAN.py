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
from scipy import stats

from load import load_wind
from load import load_solar
import plots
import wandb
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from wandb.keras import WandbCallback

import numpy as np


class DE_GAN():
    def __init__(self):
        # Input shape
        self.img_rows = 1
        self.img_cols = 24
        self.num_channels = 1
        self.img_shape = (self.img_rows, self.img_cols, self.num_channels)
        self.latent_dim = 100
        self.mean = 0
        self.std = 0.5
        self.learn_rate = 0.0001
        self.g_losses, self.d_losses, self.metric_l = [], [], []
        self.num_classes = 4
        self.ks_stats, self.ks_pvals = [[] for _ in range(self.num_classes)], [[] for _ in range(self.num_classes)]
        self.t_stats, self.t_pvals = [[] for _ in range(self.num_classes)], [[] for _ in range(self.num_classes)]
        self.ks_stats_summ, self.ks_pvals_summ, self.t_stats_summ, self.t_pvals_summ = [], [], [], []
        self.title = 'solar/real bi'

        optimizer = Adam(self.learn_rate, 0.2)

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

        img = self.generator([noise])

        # For the combined model we will only train the generator
        self.discriminator.trainable = False

        # The discriminator takes generated image as input and determines validity
        # and the label of that image
        valid = self.discriminator([img])

        # The combined model  (stacked generator and discriminator)
        # Trains generator to fool discriminator
        self.combined = Model([noise], valid)
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
        model.add(Dense(np.prod(self.img_shape, ), activation='sigmoid'))
        model.add(Reshape(self.img_shape))

        noise = Input(shape=(self.latent_dim,))
        validity = model(noise)
        print('g noise ', noise.shape)
        print('g validity ', validity.shape)

        generator = Model([noise], validity, name='generator')
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

        img = Input(shape=self.img_shape, )

        flat_img = Flatten()(img)
        validity = model(flat_img)
        print('d img ', img.shape)
        print('d flat img ', flat_img.shape)
        print('d validity ', validity.shape)

        discriminator = Model([img], validity, name='discriminator')
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

            # print(labels)
            # Generate a half batch of new images
            gen_imgs = self.generator.predict([noise])
            # print("gen imgs ", gen_imgs.shape)

            # print('final imgs ', imgs.shape)
            # print('final labels ', labels.shape)
            # print('final valid ', valid.shape)
            # Train the discriminator
            d_loss_real = self.discriminator.train_on_batch([imgs], valid)
            d_loss_fake = self.discriminator.train_on_batch([gen_imgs], fake)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            # ---------------------
            #  Train Generator
            # ---------------------
            # Train the generator
            g_loss = self.combined.train_on_batch([noise], valid)

            # Plot the progress
            metric = 100 * d_loss[1]
            print("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], metric, g_loss))
            self.g_losses.append(g_loss)
            self.d_losses.append(d_loss[0])
            self.metric_l.append(metric)
            # self.w_losses.append(w_loss)

            # Save distribution statistic
            # KS statistic
            gen_batch = gen_imgs.reshape(gen_imgs.shape[0] * gen_imgs.shape[2])
            real_batch = imgs.reshape(imgs.shape[0] * imgs.shape[2])
            (ks_stat, ks_pval) = stats.ks_2samp(real_batch[::batch_size], gen_batch[::batch_size])
            self.ks_stats_summ.append(ks_stat)
            self.ks_pvals_summ.append(ks_pval)
            # T-test statistic
            w0 = np.var(real_batch)
            w1 = np.var(gen_batch)
            var_prop = w1 / w0
            # print('Proportion 0-1: ', w1/w0)
            # print(len(real_batch[::batch_size]))
            equal_vars = False
            if 1 / 2 < var_prop < 2:
                equal_vars = True
            (t_stat, t_pval) = stats.ttest_ind(real_batch, gen_batch, equal_var=equal_vars)
            self.t_stats_summ.append(t_stat)
            self.t_pvals_summ.append(t_pval)
            # if epoch % 20 == 0:
            #     plots.record_stats(self, X_train, y_train)
            # print(self.ks_stats)
            # If at save interval => save generated image samples
            if epoch % sample_interval == 0:
                save = False
                script_dir = os.path.dirname(__file__)
                results_dir = os.path.join(script_dir,
                                           'images/Decimal GAN/%s/%d-(%0.1f,%0.1f)-noise%d-batch%d-lr%0.5f-[256,512,1024,512,512,512] sigm/' % (
                                           self.title, epochs - 1, self.mean, self.std, self.latent_dim, batch_size,
                                           self.learn_rate))
                if not os.path.isdir(results_dir):
                    os.makedirs(results_dir)
                if epoch == epochs - 1:
                    plots.stats_to_csv(self, results_dir)
                    save = True
                # Plot and save statisticis
                plots.generate_sample(self, imgs, results_dir, epoch)
                plots.plot_loss(self, True, results_dir, epochs)
                plots.distributions(self, True, imgs, gen_imgs, results_dir)
                #plots.labeled_distributions(self, True, X_train, y_train, results_dir, epoch)
                plots.save_stats(self, results_dir, self.ks_stats, self.ks_pvals, self.ks_stats_summ,
                                 self.ks_pvals_summ, 'KS', epochs, False)
                plots.save_stats(self, results_dir, self.t_stats, self.t_pvals, self.t_stats_summ, self.t_pvals_summ,
                                 'T-test', epochs, False)


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
    X, y = load_solar()
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

    print('Data loaded !')
    cgan = DE_GAN()
    cgan.train(X, y, epochs=epochs + 1, batch_size=batch_size, sample_interval=sample_interval)

    # cgan.combined.fit(X_train, y_train, validation_data=(X_test, y_test))
    # cgan.combined.fit(X_train, y_train, validation_data=(X_test, y_test), callbacks=[WandbCallback()])