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

class DE_CGAN():
    def __init__(self):
        # Input shape
        self.img_rows = 1
        self.img_cols = 24
        self.num_classes = 4
        self.num_channels = 1
        self.img_shape = (self.img_rows, self.img_cols, self.num_channels)
        self.latent_dim = 100
        self.mean = 0
        self.std = 2
        self.learn_rate = 0.0002

        self.g_losses, self.d_losses, self.metric_l = [], [], []
        self.ks_stats, self.ks_pvals = [[] for _ in range(self.num_classes)], [[] for _ in range(self.num_classes)]
        self.t_stats, self.t_pvals = [[] for _ in range(self.num_classes)], [[] for _ in range(self.num_classes)]
        self.mse = [[] for _ in range(self.num_classes)]
        # self.ssim = [[] for _ in range(self.num_classes)]
        self.std_cond = [[] for _ in range(self.num_classes)]
        self.mmd = [[] for _ in range(self.num_classes)]
        self.ks_stats_summ, self.ks_pvals_summ, self.t_stats_summ, self.t_pvals_summ, self.std_summ, self.bhat_summ, self.wasd_summ, self.mse_summ = [], [], [], [], [], [], [], []
        self.data_type = 'wind'
        self.title = self.data_type + '/bi'

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
        model.add(Dense(np.prod(self.img_shape,), activation='sigmoid'))
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
        half_batch = int(batch_size / 2)
        for epoch in range(epochs):
            # ---------------------
            #  Train Discriminator
            # ---------------------
            # Select a random half batch of images
            idx = np.random.randint(0, X_train.shape[0], half_batch)
            imgs, labels = X_train[idx], y_train[idx]
            # Sample noise as generator input
            noise = np.random.normal(self.mean, self.std, (half_batch, self.latent_dim))
            # Generate a half batch of new images
            gen_imgs = self.generator.predict([noise, labels])
            # Train the discriminator
            d_loss_real = self.discriminator.train_on_batch([imgs, labels], valid[:half_batch])
            d_loss_fake = self.discriminator.train_on_batch([gen_imgs, labels], fake[:half_batch])
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

            # Save distribution statistic
            # KS statistic
            gen_batch = gen_imgs.reshape(gen_imgs.shape[0]*gen_imgs.shape[2])
            real_batch = imgs.reshape(imgs.shape[0]*imgs.shape[2])

            (ks_stat, ks_pval) = stats.ks_2samp(real_batch, gen_batch)
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
            if epoch % 20 == 0:
                plots.record_stats(self, X_train, y_train)
            # print(self.ks_stats)
            # If at save interval => save generated image samples

            if epoch % sample_interval == 0:
                save = False
                script_dir = os.path.dirname(__file__)
                results_dir = os.path.join(script_dir, 'images/Decimal CGAN/%s/%d-(%0.1f,%0.1f)-noise%d-batch%d-lr%0.5f-[256,512,1024,512,512,512] half batch test ksall/' %(self.title, epochs-1,self.mean, self.std,self.latent_dim,batch_size,self.learn_rate))
                if not os.path.isdir(results_dir):
                    os.makedirs(results_dir)

                # Samples and loss
                plots.sample_plots(self, True, results_dir, epoch, X_train, y_train)
                plots.plot_loss(self, True, results_dir, epochs)
                #distributions
                plots.distributions(self, True, imgs, gen_imgs, results_dir, self.data_type)
                plots.labeled_distributions(self, True, X_train, y_train, results_dir, epoch, self.data_type)
                #statistics
                plots.save_stats(self, results_dir, self.ks_stats, self.ks_pvals, self.ks_stats_summ, self.ks_pvals_summ, 'KS',  epochs)
                plots.save_stats(self, results_dir, self.t_stats, self.t_pvals, self.t_stats_summ, self.t_pvals_summ, 'T-test',  epochs)
                #metrics
                plots.metrics_plots_conditioned(results_dir, self.mse, 'MSE')
                # plots.metrics_plots_conditioned(results_dir, self.ssim, 'SSIM')
                plots.metrics_plots_conditioned(results_dir, self.std_cond, 'Standard Deviation')
                # plots.metrics_plots_conditioned(results_dir, self.mmd, 'MMD')
                metrics = [self.mse_summ, self.bhat_summ, self.wasd_summ]
                labels = ['Mean Square Error','Bhattacharyya Distance','Wasserstain Distance']
                plots.metrics_plot(results_dir, metrics, labels)
                # plots.metrics_plot(results_dir, self.bhat_summ,  'Bhattacharyya')
                # plots.metrics_plot(results_dir, self.wasd_summ, 'Wasserstain Distance')
                plots.simple_plot(results_dir, self.std_summ, 'Standard Deviation Summary')
                plots.autocorrelation(results_dir, gen_imgs[0][0], imgs[0][0])

                if epoch == epochs-1:
                    # PRINT LAST METRICS FOR TABLE
                    print('Metrics output -----------------------')
                    print('MSE: ', self.mse_summ[-1])
                    print('Bhattacharya: ', self.bhat_summ[-1])
                    print('Wasserstein distance: ', self.wasd_summ[-1])
                    print('Standard deviation: ', self.std_summ[-1])
                    # plots.stats_to_csv(self, results_dir)
                    save = True

if __name__ == '__main__':
    epochs = 10000
    batch_size = 64
    learning_rate = 0.0002

    sample_interval = (epochs) / 20
    X, y = load_wind()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

    print('Data loaded !')
    cgan = DE_CGAN()
    cgan.train(X, y, epochs=epochs+1, batch_size=batch_size, sample_interval=sample_interval)

    #histor # results = cgan.combined.evaluate(X_test,y_test,verbose=0)
    # print("test loss, test acc:", results)
    #
    # # Generate predictions (probabilities -- the output of the last layer)
    # # on new data using `predict`
    # print("Generate predictions for 3 samples")
    # noise = np.random.normal(0, 2, (3, 300))
    # predictions = cgan.combined.predict([noise,y_test[:3]])
    # plots.sample_plots(cgan, True, '', 0, X_test, y_test)
    # print("predictions shape:", predictions.shape)
    # cgan.combined.fit(X_train, y_train, validation_data=(X_test, y_test))