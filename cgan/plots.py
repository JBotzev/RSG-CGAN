import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import pandas as pd
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import mean_squared_error
import math
from scipy.stats import wasserstein_distance

import statistics
from keras.utils.vis_utils import plot_model
import os


def sample_plots(self, save, dir, epoch, X_train, y_train, use_decimal=True):
    noise = np.random.normal(self.mean, self.std, (4,self.latent_dim))
    decimal_labels = np.arange(0, 4).reshape(-1, 1)
    if use_decimal:
        labels = np.arange(0, 4).reshape(-1, 1)
    else:
        labels = np.eye(4)
    # bi_labels = np.array([[1,0,0],[0,1,1],[0,0,1],[0,1,0]])
    # bi_labels = bi_labels.reshape(4,self.num_classes)
    # print('bi labels plot ', bi_labels.shape, bi_labels )
    # print('noise ',noise.shape,noise)
    # print('sampled labels ',sampled_labels.shape, sampled_labels)
    gen_imgs = self.generator.predict([noise, labels])
    # print('GENERATED IMAGE', gen_imgs.shape, gen_imgs)

    r, c = (2, 2)
    fig, axis = plt.subplots(r, c, sharex = True, sharey = True)
    cnt = 0
    for i in range(r):
        for j in range(c):
            for _ in range(X_train.shape[0]):
                idx = np.random.randint(0, X_train.shape[0])
                if y_train[idx] == cnt:
                    imgs = X_train[idx]
                    break
            axis[i,j].plot(gen_imgs[cnt][0])
            axis[i,j].plot(imgs[0])
            axis[i,j].set_title("Condition: %d" % decimal_labels[cnt])
            # axis[i,j].ylim([0,1])
            axis[i,j].legend(['Generated', 'Real'])
            axis[i,j].set_xlabel('Hours')
            axis[i,j].set_ylabel('MW Power Generated')
            cnt += 1
    # plt.xlabel('Hours')
    # plt.ylabel('MW Power Generated')
    fig.tight_layout()

    if save:
        plt.savefig(dir + "Samples %d.png" %epoch)
        self.combined.save(dir + 'model.h5')
    plt.close()


def plot_loss(self, save, dir, epoch):
    plt.figure()
    plt.plot(moving_average(self.g_losses))
    plt.plot(moving_average(self.d_losses))
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.title("Loss plot")
    plt.legend(['Generator loss','Discriminator loss'])

    if save:
        plt.savefig(dir + "Loss %d.png"%epoch)
    # plt.savefig("images/%d" % epoch)
    # plt.show()
    plt.close()
    plt.figure()
    plt.plot(moving_average(self.metric_l,100))
    plt.xlabel('Iterations')
    plt.ylabel('Accuracy')
    plt.title("Discriminator Accuracy")
    if save:
        plt.savefig(dir + "Metric %d.png"%epoch)
    plt.close()

# CHECK AND FIX SIZE OF T TEST SUMMARY < 30
def distributions(self,save, imgs, gen_imgs, dir, title):
    plt.figure()
    gen_imgs = gen_imgs.reshape((gen_imgs.shape[0]*gen_imgs.shape[2]))
    imgs = imgs.reshape((imgs.shape[0]*imgs.shape[2]))

    # SUMMARY METRICS ==============================================================================
    #STD
    self.std_summ.append(np.std(gen_imgs))
    #BATCHARAYA
    self.bhat_summ.append(-bhattacharyya(gen_imgs, imgs))
    #Wasserstein D
    self.wasd_summ.append(wasserstein_distance(gen_imgs, imgs))
    #MSE
    self.mse_summ.append(mean_squared_error(gen_imgs, imgs))
    # print('new shape ', gen_imgs.shape)
    sns.kdeplot(gen_imgs)
    sns.kdeplot(imgs)
    # sns.displot(imgs[0], kde='True')
    plt.title("Batch Distribution")
    plt.legend(['Generated','Real'])
    if save:
        plt.savefig(dir + "Batch Distribution.png")
    # plt.show()

    plt.figure()
    sns.kdeplot(gen_imgs,cumulative=True)
    sns.kdeplot(imgs,cumulative=True)
    plt.xlabel(title + '(MW)')
    plt.ylabel('CDF')
    plt.title(title + "Batch CDF")
    plt.legend(['Generated', 'Real'])
    plt.close()


def labeled_distributions(self,save, X_train, y_train, dir, epoch, title):
    batch_size = 32
    # print('LABELED DISTRIBUTIONS')
    # print(X_train.shape[0])
    gen_batch_l = []
    real_batch_l = []
    for i in range(self.num_classes):
        real_batch = []
        labels = np.full(shape=batch_size,fill_value=i,dtype=np.int)

        for k in range(batch_size):
            for _ in range(X_train.shape[0]):
                idx = np.random.randint(0, X_train.shape[0])
                if y_train[idx] == i:
                    real_batch.append(X_train[idx])
                    break

        real_batch = np.array(real_batch)
        # print('Labels %d: '%i, ' ', real_batch.shape)
        noise = np.random.normal(self.mean, self.std, (batch_size, self.latent_dim))
        gen_batch = self.generator.predict([noise, labels])
        gen_batch_arr = gen_batch.reshape(gen_batch.shape[0]*gen_batch.shape[2])
        real_batch_arr = real_batch.reshape(real_batch.shape[0]*real_batch.shape[2])
        gen_batch_l.append(gen_batch_arr)
        real_batch_l.append(real_batch_arr)

    # LABEL DISTRIBUTIONS
    r, c = (2, 2)
    fig, axis = plt.subplots(r, c)
    cnt = 0
    for k in range(r):
        for j in range(c):
            sns.kdeplot(gen_batch_l[cnt], ax=axis[k][j])
            sns.kdeplot(real_batch_l[cnt], ax=axis[k][j])
            axis[k,j].set_title("Condition: %d" % cnt)
            axis[k,j].legend(['Generated','Real'])
            cnt += 1
    plt.xlabel(title + '(MW)')
    fig.suptitle('Distributions')
    fig.tight_layout()

    if save:
        plt.savefig(dir + "Distributions %d.png"%(epoch))

    # CDF
    r, c = (2, 2)
    figa, axis1 = plt.subplots(r, c)
    cnt = 0
    for k in range(r):
        for j in range(c):
            sns.kdeplot(gen_batch_l[cnt], ax=axis1[k][j], cumulative=True)
            sns.kdeplot(real_batch_l[cnt], ax=axis1[k][j], cumulative=True)
            axis1[k,j].set_title("%d" % cnt)
            axis1[k,j].legend(['Generated','Real'])
            cnt += 1
    figa.suptitle('CDFs')
    figa.tight_layout()

    if save:
        plt.savefig(dir + "CDF.png")
    # plt.show()
    plt.close()


def record_stats(self, X_train, y_train):
    batch_size = 32
    noise = np.random.normal(self.mean, self.std, (batch_size, self.latent_dim))
    # print('LABELED DISTRIBUTIONS')
    # print(X_train.shape[0])
    gen_batch_l = []
    real_batch_l = []
    for i in range(self.num_classes):
        real_batch = []
        labels = np.full(shape=batch_size,fill_value=i,dtype=np.int)

        for k in range(batch_size):
            for _ in range(X_train.shape[0]):
                idx = np.random.randint(0, X_train.shape[0])
                if y_train[idx] == i:
                    real_batch.append(X_train[idx])
                    break

        real_batch = np.array(real_batch)
        gen_batch = self.generator.predict([noise, labels])
        gen_batch_arr = gen_batch.reshape(gen_batch.shape[0]*gen_batch.shape[2])
        real_batch_arr = real_batch.reshape(real_batch.shape[0]*real_batch.shape[2])
        gen_batch_l.append(gen_batch_arr)
        real_batch_l.append(real_batch_arr)

        # KS statistic
        # print(len(real_batch))
        (ks_stat, ks_pval) = stats.ks_2samp(real_batch_arr, gen_batch_arr)
        self.ks_stats[i].append(ks_stat)
        self.ks_pvals[i].append(ks_pval)
        # T-test taking every nth element where n is batch size=32
        w0 = np.var(real_batch_arr)
        w1 = np.var(gen_batch_arr)
        var_prop = w1/w0
        # print('Proportion 0-1: ', w1/w0)
        equal_vars = False
        if 1/2 < var_prop < 2:
            equal_vars = True
        (t_stat, t_pval) = stats.ttest_ind(real_batch_arr, gen_batch_arr,equal_var=equal_vars)
        self.t_stats[i].append(t_stat)
        self.t_pvals[i].append(t_pval)

        # again add list in DECGAN for each label appen
        # EVALUATION METRICS
        self.mse[i].append(mean_metrics(gen_batch,real_batch)[0])
        self.std_cond[i].append(np.std(gen_batch_arr))
        #concat batches
        # conc = np.concatenate((gen_batch_arr,real_batch_arr))
        # print(conc)
        # sigma = np.median(conc)/2
        # mmd_val = mmd(gen_batch_arr[:, None], real_batch_arr[:, None], sigma)
        # print('mmd ', mmd_val)
        # self.mmd_cond[i].append(mmd_val)
        # self.ssim[i].append(mean_metrics(gen_batch,real_batch)[1])

def mean_metrics(gen_batch, real_batch):
    # print('batch ',gen_batch.shape, gen_batch)
    mse = []
    ssim = []
    for i in range(len(gen_batch)):
        gen_img = gen_batch[i][0].reshape(24)
        real_img = real_batch[i][0]
        # print('img ',gen_img.shape, gen_img)
        # print('img r ', real_img.shape, real_img)
        mse.append(mean_squared_error(real_img, gen_img))

        #ssim.append(ssim(real_img, gen_img, data_range=gen_img.max() - gen_img.min()))

    return [np.mean(mse)]

def metrics_plots_conditioned(dir,  metric, title):
    plt.figure()
    r, c = (2, 2)
    fig, axis = plt.subplots(r, c)
    cnt = 0
    for i in range(r):
        for j in range(c):
            axis[i, j].plot(metric[cnt])
            axis[i, j].set_title("Condition: %d" % cnt)
            axis[i, j].set_xlabel('Iterations')
            axis[i, j].set_ylabel(title, color='blue')
            cnt += 1
    fig.suptitle(title)
    fig.tight_layout()
    plt.savefig(dir + "%s.png" % (title))
    plt.close()

def metrics_plot(dir, metrics, labels):
    fig, axis = plt.subplots()
    axis2 = axis.twinx()
    for i in range(len(metrics)):
        if i == 1:
            axis.plot(metrics[i], label=labels[i], color='g')
            axis.set_ylabel('Bhattacharya Values',color='g')
        else:
            axis2.plot(metrics[i], label=labels[i])
            axis2.set_ylabel('Metric Values')
    plt.xlabel('Iteration')
    plt.title('Batch Metrics')
    fig.legend()
    plt.savefig(dir + "Batch Metrics.png")

def save_stats(self, directory, stats, pvals, stats_summ, pvals_summ, title, epoch, save_labeled=True):
    if save_labeled:
        plt.figure()
        r, c = (2, 2)
        fig, axis = plt.subplots(r, c)
        cnt = 0
        for i in range(r):
            for j in range(c):
                axis2 = axis[i, j].twinx()
                axis[i, j].plot(moving_average(stats[cnt]), color='blue', label='Statistic',)
                axis2.plot(moving_average(pvals[cnt]), color='orange', label='P value', alpha=0.7)
                axis[i, j].set_title("Condition: %d" % cnt)
                axis[i, j].set_xlabel('Iterations')
                axis[i, j].set_ylabel('Statistic', color='blue')
                axis2.set_ylabel('P value',color='orange')
                cnt += 1
        fig.suptitle(title)
        fig.tight_layout()
        plt.savefig(directory + "Statistics %s %d.png"%(title,epoch))
        plt.close()

    fig, ax = plt.subplots()
    ax2 = ax.twinx()
    ax.plot(moving_average(stats_summ,100))
    ax2.plot(moving_average(pvals_summ,100), color="orange", label='P value', alpha=0.7)

    plt.title("%s statistic (batch)" % title)
    fig.legend(['Statistic','P value'],loc='upper left')
    ax.set_xlabel('Iterations')
    ax.set_ylabel('Statistic')
    ax2.set_ylabel('P value')
    plt.savefig(directory + "Batch statistics %s.png"%(title))
    plt.close('all')

def stats_to_csv(self, results_dir):
    print('Saving statistics . . .')
    print('ks pvals length ',len(self.ks_pvals_summ))
    print('t pvals length ', len(self.t_pvals_summ))
    df = pd.DataFrame({"P-values": np.array(self.ks_pvals_summ), "Statistic": np.array(self.ks_stats_summ)})
    df.to_csv(results_dir + "KS test.csv", index=False)
    df = pd.DataFrame(
        {"P-values": np.array(self.t_pvals_summ), "Statistic": np.array(self.t_stats_summ)})
    df.to_csv(results_dir + "T test.csv", index=False)

def generate_sample(self, imgs, dir,epoch):
    noise = np.random.normal(self.mean, self.std, (1, self.latent_dim))
    gen_imgs = self.generator.predict([noise])
    # print(gen_imgs)
    # print(gen_imgs.shape)
    print(imgs[0])
    print(imgs.shape)
    plt.plot(gen_imgs[0][0])
    plt.plot(imgs[0][0])
    plt.xlabel('Hours')
    plt.ylabel('MW Power Generated')
    plt.legend(['Generated','Real'])
    plt.title("Generated Sample")
    plt.savefig(dir + "Sample %d.png" % epoch)

def simple_plot(dir, array, title):
    plt.figure()
    plt.plot(array)
    plt.xlabel('Iterations')
    plt.ylabel('Value')
    plt.title(title)
    plt.savefig(dir + title + ".png")

def autocorrelation(dir, gen, real):
    plt.figure()
    print(gen.shape)
    gen = gen.reshape(24)
    print(real.shape)
    # auto_corr_gen = np.correlate(gen, gen, mode="full")
    # auto_corr_real = np.correlate(real, real, mode="full")
    # Creating Autocorrelation plot
    auto_corr_gen = acf(gen)
    auto_corr_real = acf(real)

    plt.plot(auto_corr_gen)
    plt.plot(auto_corr_real)
    plt.xlabel('Time (Hours)')
    plt.ylabel('Auto-corr Coefficients')
    plt.legend(['Generated','Real'])
    plt.title("Autocorrelation")
    plt.savefig(dir + "Autocorrelation.png")


def acf(x):
    return np.array([1]+[np.corrcoef(x[:-i], x[i:])[0,1]  \
        for i in range(1, len(x)-1)])

def moving_average(x, w=50):
    return np.convolve(x, np.ones(w), 'valid') / w

def bhattacharyya(a, b):
    """ Bhattacharyya distance between distributions (lists of floats). """
    if not len(a) == len(b):
        raise ValueError("a and b must be of the same size")
    return -math.log(sum((math.sqrt(u * w) for u, w in zip(a, b))))