import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
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

    if not os.path.isdir(dir) and save:
        os.makedirs(dir)
    if save:
        plt.savefig(dir + "Samples %d.png" %epoch)
    # fig.savefig("images/%d/Samples.png" % epoch)
    plt.close()

def plot_loss(self, save, dir, epoch):
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
    print(stats.ks_2samp(gen_imgs, imgs))
    print(stats.ttest_ind(gen_imgs, imgs))

def labeled_distributions(self,save, X_train, y_train, dir, epoch):
    batch_size = 32
    noise = np.random.normal(self.mean, self.std, (batch_size, self.latent_dim))
    # print('LABELED DISTRIBUTIONS')
    # print(X_train.shape[0])
    for i in range(self.num_classes):
        real_batch = []
        labels = np.full(shape=batch_size,fill_value=i,dtype=np.int)

        for k in range(batch_size):
            for _ in range(X_train.shape[0]):
                idx = np.random.randint(0, X_train.shape[0])
                if y_train[idx] == i:
                    real_batch.append(X_train[idx])
                    break;


        real_batch = np.array(real_batch)
        m = np.ndarray.max(real_batch)
        min = np.ndarray.min(real_batch)
        mean = np.ndarray.mean(real_batch)
        # print("Maximum value of wind", m)
        # print("Minimum value of wind", min)
        # print("Mean value of wind", mean)
        # print('Labels %d: '%i, ' ', real_batch.shape)
        gen_batch = self.generator.predict([noise, labels])
        gen_batch = gen_batch.reshape((gen_batch.shape[0]*gen_batch.shape[2]))
        real_batch = real_batch.reshape(real_batch.shape[0]*real_batch.shape[2])
        # print(real_batch)

        # r, c = (2, 2)
        # fig, axis = plt.subplots(r, c, sharex = True, sharey = True)
        # cnt = 0
        # for i in range(r):
        #     for j in range(c):
        #         axis[i,j].plot(gen_imgs[cnt][0])
        #         axis[i,j].set_title("Wind: %d" % i)
        #         # axis[i,j].ylim([0,1])
        #         # axis[i,j].set_xlabel('Hours')
        #         # axis[i,j].set_ylabel('MW Power Generated')
        #         cnt += 1
        # # plt.xlabel('Hours')
        # # plt.ylabel('MW Power Generated')
        # fig.tight_layout()
        sns.kdeplot(gen_batch)
        sns.kdeplot(real_batch)
        plt.title("Distributions label %d"%i)
        plt.legend(['Generated','Real'])
        if save:
            plt.savefig(dir + "Distributions %d.png"%(i))
        # plt.show()
        plt.close()
        print(stats.ks_2samp(real_batch, gen_batch))
        print(stats.ttest_ind(real_batch, gen_batch))