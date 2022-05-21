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
from tensorflow.python.keras.backend import dot
from tensorflow.python.keras.layers import concatenate

from load import load_wind, load_solar
import plots
import f_plots
import wandb
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from wandb.keras import WandbCallback

import numpy as np

class Forecast_de_CGAN():
    def __init__(self):
        # Input shape
        self.img_rows = 1
        self.img_cols = 24
        self.num_classes = 4
        self.num_channels = 1
        self.img_shape = (self.img_rows, self.img_cols, self.num_channels)
        self.mean = 0.05
        self.std = 1
        self.learn_rate = 0.0005
        self.g_losses, self.d_losses, self.metric_l = [], [], []
        self.ks_stats, self.ks_pvals = [[] for _ in range(self.num_classes)], [[] for _ in range(self.num_classes)]
        self.t_stats, self.t_pvals = [[] for _ in range(self.num_classes)], [[] for _ in range(self.num_classes)]
        self.ks_stats_summ, self.ks_pvals_summ, self.t_stats_summ, self.t_pvals_summ = [], [], [], []
        self.title = 'wind/bi'
        self.training_days = 3  # number of previous days + the current one (for real cannot be 1)
        self.real_noise = False  # use real data for previous days
        if self.real_noise:
            self.latent_dim = (self.training_days-1) * 24
        else:
            self.latent_dim = 100

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
        label = Input(shape=(self.training_days,))
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
        # model = Sequential()
        # # n_nodes = self.img_rows * self.img_cols * self.latent_dim
        # model.add(Dense(256, input_dim=self.latent_dim+self.training_days))
        # model.add(LeakyReLU(alpha=0.2))
        # model.add(BatchNormalization(momentum=0.8))
        # model.add(Dense(512))
        # model.add(LeakyReLU(alpha=0.2))
        # model.add(BatchNormalization(momentum=0.8))
        # model.add(Dense(1024))
        # model.add(LeakyReLU(alpha=0.2))
        # model.add(BatchNormalization(momentum=0.8))
        # model.add(Dense(np.prod(self.img_shape,), activation='tanh'))
        # model.add(Reshape(self.img_shape))
        #
        # noise = Input(shape=(self.latent_dim,))
        # label = Input(shape=(self.training_days,))
        # # label_embedding = Flatten()(Embedding(self.num_classes, self.latent_dim)(label))
        # # model_input = multiply([noise, label_embedding])
        # model_input = concatenate([noise, label], axis=1)

        noise = Input(shape=(self.latent_dim,),)
        label = Input(shape=(self.training_days,),)

        x = concatenate([noise, label])
        print('x = img ', x.shape)
        print('label ', label.shape)
        print('y ', y.shape)

        x = Dense(256)(x)
        x = LeakyReLU(alpha=0.2)(x)
        x = BatchNormalization(momentum=0.8)(x)
        x = Dense(512)(x)
        x = LeakyReLU(alpha=0.2)(x)
        x = BatchNormalization(momentum=0.8)(x)
        x = Dense(1024)(x)
        x = LeakyReLU(alpha=0.2)(x)
        x = BatchNormalization(momentum=0.8)(x)
        x = Dense(np.prod(self.img_shape, ), activation='tanh')(x)
        x = Reshape(self.img_shape)(x)
        #new
        # label_embedding = Flatten()(Embedding(self.num_classes, self.latent_dim, input_length=self.training_days)(label))
        # print('label emb ', label_embedding)# output shape (batch_size, self.label_dim, self.latent_dim)
        # model_input = multiply([noise, label_embedding])  # output shape (batch_size, self.label_dim, self.latent_dim)
        print('g model_input ', x.shape)
        # model_input = dot(model_input, axes=1)  # output shape (batch_size, self.latent_dim * self.label_dim)
        # print('g model_input ', model_input.shape)
        # validity = model(model_input)
        # print('g noise ', noise.shape)
        # print('g label ', label.shape)
        # print('g label emb ', label_embedding.shape)
        #
        # print('g validity ', validity.shape)

        generator = Model([noise, label], x, name='generator')
        generator.summary()
        return generator

    def build_discriminator(self):
        img = Input(shape=self.img_shape,)
        label = Input(shape=(self.training_days,), dtype='int32')
        x = img
        y = Dense(np.prod(self.img_shape))(label)
        print('y ', y.shape)
        y = Reshape(self.img_shape)(y)
        print('x = img ', x.shape)
        print('label ', label.shape)
        print('y ', y.shape)
        x = concatenate([x, y])
        x = Dense(512)(x)
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
        y_train_td = y_train.reshape(-1, self.training_days)
        print(y_train_td.shape)
        # Adversarial ground truths
        valid = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))
        half_batch = int(batch_size / 2)
        for epoch in range(epochs):
            # ---------------------
            #  Train Discriminator
            # ---------------------
            # Select a random half batch of images
            idx = np.random.randint(0, X_train.shape[0], half_batch)
            imgs = X_train[idx]
            if self.real_noise:
                noise = self.getNoise(idx, X_train)
            else:
                noise = np.random.normal(self.mean, self.std, (half_batch, self.latent_dim))

            # sns.kdeplot(noise[0])
            # plt.show()
            label_list = []
            for i in reversed(range(self.training_days)):
                label_list.append(y_train[idx-i])
            #     print('idx-i ', idx-i)
            # print('list size ',len(label_list))
            multi_labels = np.array(label_list).reshape(half_batch, self.training_days)
            # print(labels.shape)
            # print(multi_labels.shape)
            # Generate a half batch of new images
            gen_imgs = self.generator.predict([noise, multi_labels])
            # print("gen imgs ", gen_imgs.shape)

            # print('final imgs ', imgs.shape)
            # print('final labels ', labels.shape)
            # print('final valid ', valid.shape)
            # Train the discriminator
            d_loss_real = self.discriminator.train_on_batch([imgs, multi_labels], valid[:half_batch])
            d_loss_fake = self.discriminator.train_on_batch([gen_imgs, multi_labels], fake[:half_batch])
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            # ---------------------
            #  Train Generator
            # ---------------------
            # Condition on labels
            sampled_labels = []
            for i in range(self.training_days):
                sampled_labels.append(np.random.randint(0, 4, batch_size))
            sampled_labels = np.array(sampled_labels).reshape(batch_size, self.training_days)
            # print(sampled_labels.shape)
            # print(sampled_labels)
            # Train the generator
            noise = np.random.normal(self.mean, self.std, (batch_size, self.latent_dim))
            g_loss = self.combined.train_on_batch([noise, sampled_labels], valid)

            # Plot the progress
            metric = 100*d_loss[1]
            print ("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], metric, g_loss))
            self.g_losses.append(g_loss)
            self.d_losses.append(d_loss[0])
            self.metric_l.append(metric)
            # self.w_losses.append(w_loss)

            # print(gen_imgs.shape)
            # print(imgs.shape)
            # Save distribution statistic
            # KS statistic
            gen_batch = gen_imgs.reshape(gen_imgs.shape[0]*gen_imgs.shape[2])
            real_batch = imgs.reshape(imgs.shape[0]*imgs.shape[2])
            # print('size ', len(real_batch))
            # print('size reduced', len(real_batch[::5]))
            (ks_stat, ks_pval) = stats.ks_2samp(real_batch[::5], gen_batch[::5])
            self.ks_stats_summ.append(ks_stat)
            self.ks_pvals_summ.append(ks_pval)
            # T-test statistic
            w0 = np.var(real_batch)
            w1 = np.var(gen_batch)
            var_prop = w1/w0
            # print('Proportion 0-1: ', w1/w0)
            # print(len(real_batch[::batch_size]))
            equal_vars = False
            if 1/2 < var_prop < 2:
                equal_vars = True
            (t_stat, t_pval) = stats.ttest_ind(real_batch, gen_batch,equal_var=equal_vars)
            self.t_stats_summ.append(t_stat)
            self.t_pvals_summ.append(t_pval)
            # if epoch % 20 == 0:
            #     plots.record_stats(self, X_train, y_train)
            # print(self.ks_stats)
            # If at save interval => save generated image samples
            if epoch % sample_interval == 0:
                print('Saving plots and data . . .')
                save = False
                script_dir = os.path.dirname(__file__)
                results_dir = os.path.join(script_dir, 'images/Forecast de_CGAN/%s/%d-(%0.1f,%0.1f)-noise%d real(%s)-batch%d-lr%0.4f-days%d-[256,512,1024,512,512,512] half batch test/' %(self.title, epochs-1,self.mean, self.std,self.latent_dim, self.real_noise, batch_size,self.learn_rate,self.training_days))
                if not os.path.isdir(results_dir):
                    os.makedirs(results_dir)
                if epoch == epochs-1:
                    plots.stats_to_csv(self, results_dir)
                    save = True

                # Plot and save statisticis
                f_plots.forecast_samples(self, X_train, y_train, results_dir, epoch)
                #plots.sample_plots(self, True, results_dir, epoch,X_train,y_train)
                # plots.labeled_distributions(self, True, X_train, y_train, results_dir, epoch)
                plots.plot_loss(self, True, results_dir, epochs)
                plots.distributions(self, True, imgs, gen_imgs, results_dir)
                f_plots.plot_summary_tests(results_dir, self.ks_stats_summ, self.ks_pvals_summ, 'KS',  epochs)
                f_plots.plot_summary_tests(results_dir, self.t_stats_summ, self.t_pvals_summ, 'T-test',  epochs)

    def getNoise(self, idx, X_train):
        prev_days = []
        for i in range(self.training_days - 1, 0, -1):
            prev_days.append(X_train[idx - i])
            # prev days = []
        prev_days = np.array(prev_days)
        # print('training days ', prev_days.shape[0])
        # print('batch ',prev_days.shape[1])
        # print('prev days 1', prev_days[0])
        merged = np.column_stack(prev_days)
        # print('merged ',merged.shape,merged[1])
        # print('actual', X_train[idx[1]])
        noise = []
        for i in range(batch_size):
            noise.append(np.concatenate(merged[i]))
        noise = np.array(noise)
            # print('noise ', noise.shape)
            # print("idx shape: ", idx.shape, idx)
            # print("imgs shape: ", imgs.shape)
            # print("labels shape: ", labels.shape)
            # Sample noise as generator input

        return noise


if __name__ == '__main__':
    epochs = 10000
    batch_size = 64
    sample_interval = (epochs) / 20
    X, y = load_wind()
    cgan = Forecast_de_CGAN()
    cgan.train(X, y, epochs=epochs+1, batch_size=batch_size, sample_interval=sample_interval)
    print("Training completed !")
    #cgan.combined.fit(X_train, y_train, validation_data=(X_test, y_test), callbacks=[WandbCallback()])