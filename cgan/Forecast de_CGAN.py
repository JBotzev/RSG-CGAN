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
from keras.applications.inception_v3 import InceptionV3
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
        self.mean = 0
        self.std = 2
        self.data_std = 0
        self.learn_rate = 0.0002
        self.g_losses, self.d_losses, self.metric_l = [], [], []
        self.ks_stats, self.ks_pvals = [[] for _ in range(self.num_classes)], [[] for _ in range(self.num_classes)]
        self.t_stats, self.t_pvals = [[] for _ in range(self.num_classes)], [[] for _ in range(self.num_classes)]
        self.ks_stats_summ, self.ks_pvals_summ, self.t_stats_summ, self.t_pvals_summ = [], [], [], []
        # total metrics
        self.ks_stats_summ, self.ks_pvals_summ, self.t_stats_summ, \
        self.t_pvals_summ, self.std_summ, self.std_real, self.bhat_summ, self.wasd_summ, \
        self.mse_summ, self.fid, self.is_avg, self.is_std, \
        self.is_avgR, self.is_stdR = [], [], [], [], [], [], [], [], [], [], [], [], [], []
        self.data_std = 0
        self.corr_diff = []
        self.data_type = 'solar'
        self.title = self.data_type + '/bi'
        self.activation = 'sigmoid'
        self.test = True

        #conditions and noise
        self.training_days = 7  # number of previous days + the current one (for real cannot be 1)
        self.real_noise = False  # use real data for previous days
        if self.real_noise:
            self.latent_dim = (self.training_days-1) * 24
        else:
            self.latent_dim = 300

        optimizer = Adam(self.learn_rate, 0.5)
        self.iv3 = InceptionV3(include_top=False, pooling='avg', input_shape=(299, 299, 3))

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

        x = concatenate([noise, label],axis=1)
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
        x = Dense(np.prod(self.img_shape,))(x)
        x = Activation(self.activation)(x)
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
        x = Dense(1)(x)
        x = Activation('sigmoid')(x)

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

    def train(self, X_train, y_train, X_test, y_test, epochs, batch_size=128, sample_interval=50):
        print('X TRAIN')
        print(X_train.shape)
        print('X TEST')
        print(X_test.shape)
        # Configure input
        X_train_real = np.expand_dims(X_train, axis=1)
        # y_train_td = y_train.reshape(-1, self.training_days)
        y_train_real = y_train
        X_test = np.expand_dims(X_test, axis=1)
        y_test = y_test.reshape(-1, 1)
        # Adversarial ground truths
        valid = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))
        # half_batch = int(batch_size / 2)
        if self.test:
            self.data_std = np.std(X_test)
        else:
            self.data_std = np.std(X_train)
        for epoch in range(epochs):
            X_train = X_train_real
            y_train = y_train_real
            # ---------------------
            #  Train Discriminator
            # ---------------------
            # Select a random half batch of images
            idx = np.random.randint(0, X_train.shape[0], batch_size)
            imgs = X_train[idx]
            if self.real_noise:
                noise = self.getNoise(idx, X_train)
            else:
                noise = np.random.normal(self.mean, self.std, (batch_size, self.latent_dim))

            label_list = []
            for i in reversed(range(self.training_days)):
                label_list.append(y_train[idx-i])
            multi_labels = np.array(label_list).reshape(batch_size, self.training_days)
            # Generate a half batch of new images
            gen_imgs = self.generator.predict([noise, multi_labels])
            # Train the discriminator
            d_loss_real = self.discriminator.train_on_batch([imgs, multi_labels], valid)
            d_loss_fake = self.discriminator.train_on_batch([gen_imgs, multi_labels], fake)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
            # ---------------------
            #  Train Generator
            # ---------------------
            # Condition on labels
            idx = np.random.randint(0, X_train.shape[0], batch_size)
            label_list = []
            for i in reversed(range(self.training_days)):
                label_list.append(y_train[idx - i])

            sample_multi_labels = np.array(label_list).reshape(batch_size, self.training_days)
            # Train the generator
            # print('ml 1 ', multi_labels)
            noise = np.random.normal(self.mean, self.std, (batch_size, self.latent_dim))
            g_loss = self.combined.train_on_batch([noise, sample_multi_labels], valid)

            # Plot the progress
            metric = 100*d_loss[1]
            print ("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], metric, g_loss))
            self.g_losses.append(g_loss)
            self.d_losses.append(d_loss[0])
            self.metric_l.append(metric)

            # Save distribution statistic
            # KS statistic
            new_batch_size = 32
            if epoch % 20 == 0:
                if self.test:
                    X_train = X_test
                    y_train = y_test
                    idx = np.random.randint(0, X_train.shape[0], new_batch_size)
                    if self.real_noise:
                        noise_z = self.getNoise(idx, X_train)
                    else:
                        noise_z = np.random.normal(self.mean, self.std, (new_batch_size, self.latent_dim))
                    label_list_z = []
                    for i in reversed(range(self.training_days)):
                        label_list_z.append(y_train[idx - i])
                    multi_labels_z = np.array(label_list_z).reshape(new_batch_size, self.training_days)
                    imgs = X_train[idx]
                    gen_imgs = self.generator.predict([noise_z, multi_labels_z])
                plots.record_summ_stats(self, imgs[:new_batch_size], gen_imgs[:new_batch_size])

            # If at save interval => save generated image samples
            if epoch % sample_interval == 0:
                print('Saving plots and data . . .')
                script_dir = os.path.dirname(__file__)
                results_dir = os.path.join(script_dir, 'images/Forecast de_CGAN/%s/%d-(%0.1f,%0.1f)-noise%d real(%s)-batch%d-lr%0.4f-days%d-%s-test %s x2 fixed batch size/'
                %(self.title, epochs-1,self.mean, self.std,self.latent_dim, self.real_noise, batch_size,self.learn_rate,self.training_days, self.activation,self.test))
                if not os.path.isdir(results_dir):
                    os.makedirs(results_dir)

                # GENERATE IMAGES FOR TESTING
                if self.test:
                    X_train = X_test
                    y_train = y_test
                    print('test ',X_train.shape)
                    idx = np.random.randint(0, X_train.shape[0], new_batch_size)
                    if self.real_noise:
                        noise_z = self.getNoise(idx, X_train)
                    else:
                        noise_z = np.random.normal(self.mean, self.std, (new_batch_size, self.latent_dim))
                    label_list_z = []
                    for i in reversed(range(self.training_days)):
                        print(y_train[idx-i])
                        label_list_z.append(y_train[idx - i])
                    multi_labels_z = np.array(label_list_z).reshape(new_batch_size, self.training_days)
                    imgs = X_train[idx]
                    gen_imgs = self.generator.predict([noise_z, multi_labels_z])
                    print('SAMPLE INTERVAL ', imgs.shape, ' ',gen_imgs.shape)
                # STANDARD PLOTS
                if self.training_days > 1:
                    f_plots.forecast_samples(self, X_train, y_train, results_dir, epoch)
                else:
                    plots.sample_plots(self, True, results_dir, epoch, X_train, y_train)
                imgs = imgs[:new_batch_size]
                gen_imgs = gen_imgs[:new_batch_size]
                plots.plot_loss(self, True, results_dir, epochs)
                print('DISTRIBUTION IMGS ', imgs.shape)
                print('GEN imgs ', gen_imgs.shape)
                plots.distributions(self, True, imgs, gen_imgs, results_dir, self.data_type,epoch)
                # STATISTICS
                f_plots.plot_summary_tests(results_dir, self.ks_stats_summ, self.ks_pvals_summ, 'KS',  epochs)
                f_plots.plot_summary_tests(results_dir, self.t_stats_summ, self.t_pvals_summ, 'T-test',  epochs)
                # METRICS SUMMARY
                f_plots.is_fid(self, X_train, y_train, self.iv3)
                metrics = [self.mse_summ, self.wasd_summ, self.fid]
                labels = ['Mean Square Error', 'Wasserstain Distance', 'FID']
                plots.metrics_plot(results_dir, metrics, labels, 'Batch Metrics', True)
                # OTHER
                plots.simple_plot(self, results_dir, self.std_summ, self.std_real, 'Standard Deviation Summary')
                plots.autocorrelation(results_dir, gen_imgs, imgs)
                # FID and IS
                #in_score = [self.is_avg, self.is_std, self.is_avgR, self.is_stdR]
                # labels_IS = ['IS gen', 'Std gen', 'IS real', 'Std real']
                # plots.metrics_plot_IS(results_dir, in_score, 'Inception Score')

            if epoch == epochs - 1:
                # PRINT LAST METRICS FOR TABLE
                print('Metrics output -----------------------')
                ks_range = int(len(self.ks_pvals_summ) / 10)
                ks_result = np.mean(self.ks_pvals_summ[-ks_range:-1])
                best = [np.min(self.fid), np.min(self.mse_summ), np.min(self.wasd_summ),
                        np.max(self.std_summ), np.min(self.ks_stats_summ), np.max(self.ks_pvals_summ)]
                last = [self.metric_l[-1], self.fid[-1], self.mse_summ[-1], self.wasd_summ[-1], self.std_summ[-1],
                        self.ks_stats_summ[-1], ks_result]
                # BY MSE
                index = self.mse_summ.index(np.min(self.mse_summ))
                iteration = int(index * sample_interval)
                stat_iter = int(index * sample_interval / 20)
                by_mse = [self.metric_l[iteration], self.fid[index], self.mse_summ[index], self.wasd_summ[index],
                          self.std_summ[index],
                          self.ks_stats_summ[stat_iter], self.ks_pvals_summ[stat_iter]]
                # BY FID
                index = self.fid.index(np.min(self.fid))
                iteration = int(index * sample_interval)
                stat_iter = int(index * sample_interval / 20)
                pval_record = np.mean(self.ks_pvals_summ[stat_iter - 50:stat_iter + 50])
                kstat_record = np.mean(self.ks_stats_summ[stat_iter - 50:stat_iter + 50])
                std_diff = np.abs(self.data_std - self.std_summ[index])
                by_fid = [self.metric_l[iteration], self.fid[index], self.mse_summ[index], self.wasd_summ[index],
                          self.std_summ[index], kstat_record, pval_record, self.corr_diff[index], std_diff]

                # SAVE BEST / LAST / BEST by MSE / BEST by FID
                np.savetxt(results_dir + 'Metrics best.csv', best, delimiter=",", fmt='%1.5f')
                np.savetxt(results_dir + 'Metrics last.csv', last, delimiter=",", fmt='%1.5f')
                np.savetxt(results_dir + 'Metrics by MSE.csv', by_mse, delimiter=",", fmt='%1.5f')
                np.savetxt(results_dir + 'Metrics by FID.csv', by_fid, delimiter=",", fmt='%1.5f')
                # SAVE ALL
                np.savetxt(results_dir + 'zz Accuracy.csv', self.metric_l, delimiter=",", fmt='%1.5f')
                np.savetxt(results_dir + 'zz FID.csv', self.fid, delimiter=",", fmt='%1.5f')
                np.savetxt(results_dir + 'zz MSE.csv', self.mse_summ, delimiter=",", fmt='%1.5f')
                np.savetxt(results_dir + 'zz WASD.csv', self.wasd_summ, delimiter=",", fmt='%1.5f')
                np.savetxt(results_dir + 'zz STD.csv', self.std_summ, delimiter=",", fmt='%1.5f')
                np.savetxt(results_dir + 'zz KS stat.csv', self.ks_stats_summ, delimiter=",", fmt='%1.5f')
                np.savetxt(results_dir + 'zz KS pval.csv', self.ks_pvals_summ, delimiter=",", fmt='%1.5f')
                print('Metrics output -----------------------')
                print(results_dir)

    def getNoise(self, idx, X_train):
        # print('idx ',idx.shape, idx)
        prev_days = []
        for i in range(self.training_days - 1, 0, -1):
            prev_days.append(X_train[idx - i])
        prev_days = np.array(prev_days)
        # print('prev days ', prev_days.shape)
        merged = np.column_stack(prev_days)
        noise = []
        # print('merged ',merged)
        for i in range(idx.shape[0]):
            # print('merged i:',i, ' ',merged[i])
            noise.append(np.concatenate(merged[i]))
        noise = np.array(noise)

        return noise


if __name__ == '__main__':
    epochs = 80000
    batch_size = 64
    sample_interval = (epochs) / 20
    X, y = load_solar()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42, shuffle=False, stratify=None)

    cgan = Forecast_de_CGAN()
    cgan.train(X_train, y_train, X_test, y_test, epochs=epochs+1, batch_size=batch_size, sample_interval=sample_interval)
    print("Training completed !")
