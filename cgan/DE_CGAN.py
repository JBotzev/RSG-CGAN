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
from keras.applications.inception_v3 import InceptionV3
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
        self.latent_dim = 300
        self.mean = 0
        self.std = 2
        self.learn_rate = 0.002
        self.g_losses, self.d_losses, self.metric_l = [], [], []
        self.ks_stats, self.ks_pvals = [[] for _ in range(self.num_classes)], [[] for _ in range(self.num_classes)]
        self.t_stats, self.t_pvals = [[] for _ in range(self.num_classes)], [[] for _ in range(self.num_classes)]
        #condition metrics
        self.mse = [[] for _ in range(self.num_classes)]
        self.std_cond = [[] for _ in range(self.num_classes)]
        self.mmd = [[] for _ in range(self.num_classes)]
        # total metrics
        self.ks_stats_summ, self.ks_pvals_summ, self.t_stats_summ, \
        self.t_pvals_summ, self.std_summ, self.std_real, self.bhat_summ, self.wasd_summ, \
        self.mse_summ, self.fid, self.is_avg, self.is_std, \
        self.is_avgR, self.is_stdR = [], [], [], [], [], [], [], [], [], [], [], [], [], []
        self.cond_histo_g, self.cond_histo_r = [], []
        self.data_std = 0
        self.corr_diff = []
        self.data_type = 'solar'
        self.title = self.data_type + '/bi'
        self.activation = 'sigmoid'
        self.test = True

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
        model.add(Dense(np.prod(self.img_shape,), activation=self.activation))
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

    def train(self, X_train, y_train, X_test, y_test, epochs, batch_size=128, sample_interval=50):
        print('X TRAIN')
        print(X_train.shape)
        print('X TEST')
        print(X_test.shape)
        # Configure input
        X_train_real = np.expand_dims(X_train, axis=1)
        y_train_real = y_train.reshape(-1, 1)
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
            # print('X shape ', X_train.shape)
            # ---------------------
            #  Train Discriminator
            # ---------------------
            # Select a random half batch of images
            idx = np.random.randint(0, X_train.shape[0], batch_size)
            imgs, labels = X_train[idx], y_train[idx]
            # Sample noise as generator input
            noise = np.random.normal(self.mean, self.std, (batch_size, self.latent_dim))
            # Generate a half batch of new images
            gen_imgs = self.generator.predict([noise, labels])
            # Train the discriminator
            d_loss_real = self.discriminator.train_on_batch([imgs, labels], valid)
            d_loss_fake = self.discriminator.train_on_batch([gen_imgs, labels], fake)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
            # ---------------------
            #  Train Generator
            # ---------------------
            # Condition on labels
            sampled_labels = np.random.randint(0, 4, batch_size).reshape(-1, 1)
            noise = np.random.normal(self.mean, self.std, (batch_size, self.latent_dim))
            # Train the generator
            g_loss = self.combined.train_on_batch([noise, sampled_labels], valid)
            # Plot the progress
            metric = 100*d_loss[1]
            print ("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], metric, g_loss))
            self.g_losses.append(g_loss)
            self.d_losses.append(d_loss[0])
            self.metric_l.append(metric)
            # self.w_losses.append(w_loss)

            # PASS TEST -------------------------
            # influencing summ stats, summ distribution and autocorrelation
            new_batch_size = 32
            if epoch % 20 == 0:
                if self.test:
                    idx = np.random.randint(0, X_test.shape[0], new_batch_size)
                    imgs, labels = X_test[idx], y_test[idx]
                    noise = np.random.normal(self.mean, self.std, (new_batch_size, self.latent_dim))
                    gen_imgs = self.generator.predict([noise, labels])
                    plots.record_cond_stats(self, X_test, y_test)
                else:
                    imgs, labels = imgs[:new_batch_size], labels[:new_batch_size]
                    gen_imgs = gen_imgs[:new_batch_size]
                    plots.record_cond_stats(self, X_train, y_train)

                plots.record_summ_stats(self, imgs, gen_imgs)

            # If at save interval => save generated image samples
            if epoch % sample_interval == 0:
                script_dir = os.path.dirname(__file__)
                results_dir = os.path.join(script_dir, 'images/Decimal CGAN/%s/%d-(%0.1f,%0.1f)-noise%d-batch%d-lr%0.5f-%s-test %s/'
                                           %(self.title, epochs-1,self.mean, self.std,self.latent_dim,batch_size,self.learn_rate,self.activation,self.test))
                if not os.path. isdir(results_dir):
                    os.makedirs(results_dir)

                # GENERATE IMAGES FOR TESTING
                if self.test:
                    X_train = X_test
                    y_train = y_test
                    idx = np.random.randint(0, X_test.shape[0], new_batch_size)
                    imgs, labels = X_test[idx], y_test[idx]
                    noise = np.random.normal(self.mean, self.std, (new_batch_size, self.latent_dim))
                    gen_imgs = self.generator.predict([noise, labels])
                # Samples and loss
                # PASS TEST
                print('X size: ', X_train.shape)
                plots.sample_plots(self, True, results_dir, epoch, X_train, y_train)
                plots.plot_loss(self, True, results_dir, epochs)
                #distributions
                # PASS TEST
                plots.distributions(self, True, imgs, gen_imgs, results_dir, self.data_type,epoch)
                plots.labeled_distributions(self, True, X_train, y_train, results_dir, epoch, self.data_type)
                #statistics
                plots.save_stats(self, results_dir, self.ks_stats, self.ks_pvals, self.ks_stats_summ, self.ks_pvals_summ, 'KS',  epochs)
                plots.save_stats(self, results_dir, self.t_stats, self.t_pvals, self.t_stats_summ, self.t_pvals_summ, 'T-test',  epochs)
                #METRICS CONDITIONED
                #plots.metrics_plots_conditioned(results_dir, self.mse, 'MSE')
                #plots.metrics_plots_conditioned(results_dir, self.std_cond, 'Standard Deviation')

                #METRICS SUMMARY
                # PASS TEST
                plots.is_fid(self, X_train, y_train, self.iv3)
                metrics = [self.mse_summ, self.wasd_summ, self.fid]
                labels = ['Mean Square Error','Wasserstain Distance', 'FID']
                plots.metrics_plot(results_dir, metrics, labels, 'Batch Metrics', True)
                #OTHER
                plots.simple_plot(self, results_dir, self.std_summ, self.std_real, 'Standard Deviation Summary')
                plots.autocorrelation(results_dir, gen_imgs, imgs)
                plots.plot_histogram(results_dir, self.cond_histo_g, self.cond_histo_r, epoch)
                print('Snapshot at: ', epoch)

                if epoch == epochs-1:
                    # PRINT LAST METRICS FOR TABLE
                    ks_range = int(len(self.ks_pvals_summ) / 10)
                    ks_result = np.mean(self.ks_pvals_summ[-ks_range:-1])
                    best = [np.min(self.fid), np.min(self.mse_summ), np.min(self.wasd_summ),
                            np.max(self.std_summ), np.min(self.ks_stats_summ), np.max(self.ks_pvals_summ)]
                    last = [self.metric_l[-1], self.fid[-1], self.mse_summ[-1], self.wasd_summ[-1], self.std_summ[-1],
                            self.ks_stats_summ[-1], ks_result]

                    # BY MSE
                    index = self.mse_summ.index(np.min(self.mse_summ))
                    iteration = int(index*sample_interval)
                    stat_iter = int(index*sample_interval/20)
                    by_mse = [self.metric_l[iteration], self.fid[index], self.mse_summ[index], self.wasd_summ[index], self.std_summ[index],
                            self.ks_stats_summ[stat_iter], self.ks_pvals_summ[stat_iter]]
                    # BY FID
                    index = self.fid.index(np.min(self.fid))
                    print('index ', index)
                    print('fid array ', self.fid)
                    iteration = int(index * sample_interval)
                    stat_iter = int(iteration / 20)
                    print('real pval ',self.ks_pvals_summ[stat_iter])
                    pval_record = np.mean(self.ks_pvals_summ[stat_iter-20:stat_iter+20])
                    kstat_record = np.mean(self.ks_stats_summ[stat_iter-20:stat_iter+20])
                    std_diff = np.abs(self.data_std - self.std_summ[index])
                    print('std diff ',std_diff)
                    print('corr diff ', self.corr_diff)
                    print('corr diff ', self.corr_diff[index])
                    print('FIXED PVAL ', pval_record)
                    print('stat iteration ',stat_iter)
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

if __name__ == '__main__':
    epochs = 50000
    batch_size = 64

    sample_interval = (epochs) / 20
    X, y = load_solar()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

    print('Data loaded !')
    cgan = DE_CGAN()
    cgan.train(X_train, y_train, X_test, y_test, epochs=epochs+1, batch_size=batch_size, sample_interval=sample_interval)