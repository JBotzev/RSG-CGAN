import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import os

def sample_plots(self, save, dir):
    noise = np.random.normal(self.mean, self.std, (self.num_classes,self.latent_dim))
    decimal_labels = np.arange(0, 4).reshape(-1, 1)
    ohe_labels = np.eye(4)
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

    if not os.path.isdir(dir) and save == True:
        os.makedirs(dir)
        plt.savefig(dir + "Samples.png")
    # fig.savefig("images/%d/Samples.png" % epoch)
    plt.close()

def plot_loss(self, save, dir):
    plt.plot(self.g_losses)
    plt.plot(self.d_losses)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title("Loss plot")
    plt.legend(['Generator loss','Discriminator loss'])
    if save:
        plt.savefig(dir + "Loss.png")
    # plt.savefig("images/%d" % epoch)
    plt.show()

def distributions(self,save, imgs, gen_imgs, dir):
    gen_imgs = gen_imgs.reshape((gen_imgs.shape[0]*gen_imgs.shape[2]))
    imgs = imgs.reshape((imgs.shape[0]*imgs.shape[2]))
    # print('new shape ', gen_imgs.shape)
    sns.kdeplot(gen_imgs)
    sns.kdeplot(imgs)
    # sns.displot(imgs[0], kde='True')
    plt.title("Distributions")
    plt.legend(['Generated','Real'])
    if save:
        plt.savefig(dir + "Distributions.png")
    plt.show()

    print(stats.ks_2samp(gen_imgs, imgs))
    print(stats.ttest_ind(gen_imgs, imgs))