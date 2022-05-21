from __future__ import print_function, division

import os

import pandas as pd
from keras.datasets import mnist
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, multiply
from keras.layers import BatchNormalization, Activation, Embedding, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.optimizers import RMSprop
from keras import backend
from keras.models import Sequential, Model
from keras.optimizers import Adam
from keras import metrics
from scipy import stats
from keras.initializers import RandomNormal
from tensorflow.python.keras.layers import Conv2DTranspose

from ClipConstraint import ClipConstraint
from load import load_wind
from load import load_solar
import plots
import wandb
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from wandb.keras import WandbCallback

import numpy as np

class DE_WCGAN():
    def __init__(self):
        # Input shape
        self.img_rows = 1
        self.img_cols = 24
        self.num_classes = 4
        self.num_channels = 1
        self.img_shape = (self.img_rows, self.img_cols, self.num_channels)
        self.latent_dim = 100
        self.mean = 0
        self.std = 0.02
        self.learn_rate = 0.00002
        self.g_losses, self.d_losses, self.metric_l = [], [], []
        self.ks_stats, self.ks_pvals = [[] for _ in range(self.num_classes)], [[] for _ in range(self.num_classes)]
        self.t_stats, self.t_pvals = [[] for _ in range(self.num_classes)], [[] for _ in range(self.num_classes)]
        self.ks_stats_summ, self.ks_pvals_summ, self.t_stats_summ, self.t_pvals_summ = [], [], [], []
        self.title = 'Wasserstein solar conv'
        self.n_critic = 4
        self.generator_in_channels = self.latent_dim + self.num_classes

        optimizer = RMSprop(self.learn_rate, 0.5)

        # Build and compile the discriminator
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss=[self.wasserstein_loss],
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
        self.combined.compile(loss=self.wasserstein_loss,
                              optimizer=optimizer)

    def build_generator(self):
        # weight initialization
        init = RandomNormal(stddev=0.02)
        model = Sequential()
        # n_nodes = self.img_rows * self.img_cols * self.latent_dim
        model.add(Dense(15, activation='relu', kernel_initializer='he_uniform', input_dim=self.latent_dim))
        model.add(Dense(24, activation='linear'))
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
        # weight initialization
        init = RandomNormal(stddev=0.02)
        const = ClipConstraint(0.01)
        model = Sequential()
        model.add(Dense(25, activation='relu', kernel_initializer='he_uniform', input_dim=np.prod(self.img_shape)))
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
            d_loss_temp, temp_acc = list(), list()
            for _ in range(self.n_critic):
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
                d_loss_real, acc_real = self.discriminator.train_on_batch([imgs, labels], valid)
                d_loss_fake, acc_fake = self.discriminator.train_on_batch([gen_imgs, labels], fake)
                d_loss = 0.5 * (d_loss_real + d_loss_fake)
                acc = 0.5 * (acc_real + acc_fake)
                d_loss_temp.append(d_loss)
                temp_acc.append(acc)

            self.d_losses.append(np.mean(d_loss_temp))
            self.metric_l.append(np.mean(temp_acc))
            # ---------------------
            #  Train Generator
            # ---------------------
            # Condition on labels
            sampled_labels = np.random.randint(0, 4, batch_size).reshape(-1, 1)
            # Train the generator
            noise = np.random.normal(self.mean, self.std, (batch_size, self.latent_dim))
            g_loss = self.combined.train_on_batch([noise, sampled_labels], valid)

            # Plot the progress
            print ("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, self.d_losses[epoch], 100*self.metric_l[epoch], g_loss))
            self.g_losses.append(g_loss)
            # self.w_losses.append(w_loss)

            # Save distribution statistic
            # KS statistic
            print(gen_imgs.shape)
            print(gen_imgs)
            gen_batch = gen_imgs.reshape(gen_imgs.shape[0]*gen_imgs.shape[2])
            real_batch = imgs.reshape(imgs.shape[0]*imgs.shape[2])
            (ks_stat, ks_pval) = stats.ks_2samp(real_batch, gen_batch)
            self.ks_stats_summ.append(ks_stat)
            self.ks_pvals_summ.append(ks_pval)
            # T-test statistic
            (t_stat, t_pval) = stats.ttest_ind(real_batch[::batch_size], gen_batch[::batch_size])
            self.t_stats_summ.append(t_stat)
            self.t_pvals_summ.append(t_pval)
            if epoch % 20 == 0:
                plots.record_stats(self, X_train, y_train)
            # print(self.ks_stats)
            # If at save interval => save generated image samples
            if epoch % sample_interval == 0:
                save = False
                script_dir = os.path.dirname(__file__)
                results_dir = os.path.join(script_dir, 'images/Decimal CGAN/%s/%d-(%0.1f,%0.1f)-noise%d-batch%d-lr%0.4f-[256,512,1024,512,512,512]v2/' %(self.title, epochs-1,self.mean, self.std,self.latent_dim,batch_size,self.learn_rate))
                if not os.path.isdir(results_dir):
                    os.makedirs(results_dir)
                if epoch == epochs - 1:
                    save = True
                # Plot and save statisticis
                plots.sample_plots(self, True, results_dir, epoch)
                plots.plot_loss(self, True, results_dir, epochs)
                plots.distributions(self, True, imgs, gen_imgs, results_dir)
                plots.labeled_distributions(self, True, X_train, y_train, results_dir, epoch)
                plots.save_stats(self, results_dir, self.ks_stats, self.ks_pvals, self.ks_stats_summ, self.ks_pvals_summ, 'KS',  epochs)
                plots.save_stats(self, results_dir, self.t_stats, self.t_pvals, self.t_stats_summ, self.t_pvals_summ, 'T-test',  epochs)

    # Defining Wasserstein Loss
    def wasserstein_loss(self, y_true, y_pred):
        return backend.mean(y_true * y_pred)

if __name__ == '__main__':
    # wandb.init(project="my-test-project", entity="joanbotzev")

    epochs = 10000
    batch_size = 32
    learning_rate = 0.0002
    # wandb.config = {
    #     "learning_rate": learning_rate,
    #     "epochs": epochs,
    #     "batch_size": batch_size
    # }

    sample_interval = (epochs) / 20
    X, y = load_solar()
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

    cgan = DE_WCGAN()
    cgan.train(X, y, epochs=epochs+1, batch_size=batch_size, sample_interval=sample_interval)
    #cgan.combined.fit(X_train, y_train, validation_data=(X_test, y_test), callbacks=[WandbCallback()])