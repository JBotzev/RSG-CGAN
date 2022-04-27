import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from keras.utils.vis_utils import plot_model
import os


def sample_plots(self, save, dir, epoch):
    noise = np.random.normal(self.mean, self.std, (4,self.latent_dim))

    decimal_labels = np.arange(0, 4).reshape(-1, 1)
    ohe_labels = np.eye(4)
    # bi_labels = np.array([[1,0,0],[0,1,1],[0,0,1],[0,1,0]])
    # bi_labels = bi_labels.reshape(4,self.num_classes)
    # print('bi labels plot ', bi_labels.shape, bi_labels )
    # print('noise ',noise.shape,noise)
    # print('sampled labels ',sampled_labels.shape, sampled_labels)
    gen_imgs = self.generator.predict([noise, decimal_labels])
    # print('GENERATED IMAGE', gen_imgs.shape, gen_imgs)
    r, c = (2, 2)
    fig, axis = plt.subplots(r, c, sharex = True, sharey = True)
    cnt = 0
    for i in range(r):
        for j in range(c):
            axis[i,j].plot(gen_imgs[cnt][0])
            axis[i,j].set_title("Wind: %d" % decimal_labels[cnt])
            # axis[i,j].ylim([0,1])
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
    plt.plot(self.g_losses)
    plt.plot(self.d_losses)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title("Loss plot")
    plt.legend(['Generator loss','Discriminator loss'])

    if save:
        plt.savefig(dir + "Loss %d.png"%epoch)
    # plt.savefig("images/%d" % epoch)
    # plt.show()
    plt.close()


def distributions(self,save, imgs, gen_imgs, dir, epoch):
    plt.figure()
    gen_imgs = gen_imgs.reshape((gen_imgs.shape[0]*gen_imgs.shape[2]))
    imgs = imgs.reshape((imgs.shape[0]*imgs.shape[2]))

    # print('new shape ', gen_imgs.shape)
    sns.kdeplot(gen_imgs)
    sns.kdeplot(imgs)
    # sns.displot(imgs[0], kde='True')
    plt.title("Distributions")
    plt.legend(['Generated','Real'])
    if save:
        plt.savefig(dir + "Distributions %d.png"%epoch)
    # plt.show()
    plt.close()


def labeled_distributions(self,save, X_train, y_train, dir, epoch):
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
        # print('Labels %d: '%i, ' ', real_batch.shape)
        gen_batch = self.generator.predict([noise, labels])
        gen_batch = gen_batch.reshape(gen_batch.shape[0]*gen_batch.shape[2])
        real_batch = real_batch.reshape(real_batch.shape[0]*real_batch.shape[2])
        gen_batch_l.append(gen_batch)
        real_batch_l.append(real_batch)
        # (t_stats, stats.ttest_ind(real_batch, gen_batch))
    r, c = (2, 2)
    fig, axis = plt.subplots(r, c)
    cnt = 0
    for k in range(r):
        for j in range(c):
            sns.kdeplot(gen_batch_l[cnt], ax=axis[k][j])
            sns.kdeplot(real_batch_l[cnt], ax=axis[k][j])
            axis[k,j].set_title("Wind: %d" % cnt)
            axis[k,j].legend(['Generated','Real'])
            cnt += 1
    fig.tight_layout()

    if save:
        plt.savefig(dir + "Distributions %d.png"%(epoch))
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
        gen_batch = gen_batch.reshape(gen_batch.shape[0]*gen_batch.shape[2])
        real_batch = real_batch.reshape(real_batch.shape[0]*real_batch.shape[2])
        gen_batch_l.append(gen_batch)
        real_batch_l.append(real_batch)

        # KS statistic
        (ks_stat, ks_pval) = stats.ks_2samp(real_batch, gen_batch)
        self.ks_stats[i].append(ks_stat)
        self.ks_pvals[i].append(ks_pval)
        # T-test
        (t_stat, t_pval) = stats.ttest_ind(real_batch, gen_batch)
        self.t_stats[i].append(t_stat)
        self.t_pvals[i].append(t_pval)

def save_stats(self, directory, stats, pvals, stats_summ, pvals_summ, title, epoch):
    plt.figure()
    r, c = (2, 2)
    fig, axis = plt.subplots(r, c)
    cnt = 0
    for i in range(r):
        for j in range(c):
            axis2 = axis[i, j].twinx()
            axis[i, j].plot(stats[cnt])
            axis2.plot(pvals[cnt], color='orange', label='P value', alpha=0.7)
            axis[i, j].set_title("Wind: %d" % cnt)
            fig.legend(['Statistic','P value'],loc='upper left')
            axis[i, j].set_xlabel('Iterations')
            axis[i, j].set_ylabel('Statistic')
            axis2.set_ylabel('P value')
            cnt += 1
    fig.tight_layout()
    plt.savefig(directory + "Statistics %s %d.png"%(title,epoch))
    plt.close()

    fig, ax = plt.subplots()
    ax2 = ax.twinx()
    ax.plot(stats_summ)
    ax2.plot(pvals_summ, color="orange", label='P value', alpha=0.7)

    plt.title("Summary %s statistic" % title)
    fig.legend(['Statistic','P value'],loc='upper left')
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Statistic')
    ax2.set_ylabel('P value')
    plt.savefig(directory + "Statistics Summary %s %d.png"%(title,epoch))
    plt.close('all')