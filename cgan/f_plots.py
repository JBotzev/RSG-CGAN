import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import FID
from plots import moving_average
from plots import acf

def forecast_samples(self, X_train, y_train, dir, epoch):
    idx = np.random.randint(0, X_train.shape[0])
    days = list(range(idx-self.training_days+1,idx+1))
    # print('days ',days)
    # print('idx ',idx)
    imgs = X_train[days]

    # Generate imgs
    if self.real_noise:
        noise = np.concatenate(np.concatenate(X_train[days[:-1]])).reshape(1,self.latent_dim)
    else:
        noise = np.random.normal(self.mean, self.std, (1,self.latent_dim))

    label_list = []
    for i in range(0,self.training_days):
        label_list.append(y_train[days[i]])
        print('days i ', days[i], ' ',y_train[days[i]])
    multi_labels = np.array(label_list).reshape(1, self.training_days)
    gen_imgs = self.generator.predict([noise, multi_labels]).reshape(24)

    # plt.figure()
    # plt.plot(acf(gen_imgs))
    # plt.plot(acf(imgs[self.training_days-1].reshape(24)))
    # plt.title('Autocorrelation')
    # plt.xlabel('Lags')
    # plt.ylabel('Coefficients')
    # plt.legend(['Generated', 'Real'])
    # plt.savefig(dir + "Z Autocorrelation %d.png"%epoch)
    # plt.close()
    auto_corr_gen = acf(gen_imgs)
    auto_corr_real = acf(imgs[self.training_days-1].reshape(24))
    fig, (ax1, ax2) = plt.subplots(2)
    ax2.plot(auto_corr_real)
    ax2.plot(auto_corr_gen)
    ax2.set_title('Autocorrelation')
    ax2.set_xlabel('Lags')
    ax2.set_ylabel('Coefficients')
    ax2.legend(['Real','Generated'])
    ax1.plot(imgs[self.training_days - 1].reshape(24))
    ax1.plot(gen_imgs)
    ax1.set_title('Samples')
    ax1.set_xlabel('Hours')
    ax1.set_ylabel('Generated (MW)')
    ax1.legend(['Real','Generated'])
    fig.tight_layout()
    plt.savefig(dir + "Z Autocorrelation %d.png" % epoch)
    plt.close(fig)
    # Plot
    r, c = (1, self.training_days)
    fig, axis = plt.subplots(r, c, figsize=(15,5), sharex = True, sharey = True)
    cnt = 0
    for i in range(r):
        for j in range(c):
            axis[j].plot(imgs[cnt][0],label='Real')
            if j == c-1:
                axis[j].plot(gen_imgs,label='Forecasted')
                axis[j].legend()
            # print('all labels ',y_train[days])
            # print('labels ', y_train[days[cnt]])
            axis[j].set_title("Day %d - Wind %d" % (cnt, y_train[days[cnt]]))
            # axis[i,j].ylim([0,1])
            axis[j].set_xlabel('Hours')
            axis[j].set_ylabel('MW Power Generated')
            axis[j].set_ylim([0, 1])
            cnt += 1
    fig.tight_layout()
    plt.savefig(dir + "Z Samples %d.png" % epoch)
    self.combined.save(dir + 'model.h5')
    plt.close()

def plot_summary_tests(directory,stats_summ, pvals_summ, title, epoch):
    fig, ax = plt.subplots()
    ax2 = ax.twinx()
    ax.plot(moving_average(stats_summ))
    ax2.plot(moving_average(pvals_summ), color="orange", label='P value', alpha=0.7)

    plt.title("Summary %s statistic" % title)
    fig.legend(['Statistic','P value'],loc='upper left')
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Statistic')
    ax2.set_ylabel('P value')
    plt.savefig(directory + "Statistics Summary %s %d.png"%(title,epoch))
    plt.close('all')

def is_fid(self,X_train,y_train,model):
    batch_size = 100
    idx = np.random.randint(0, X_train.shape[0], batch_size)
    if self.real_noise:
        noise_z = self.getNoise(idx, X_train)
    else:
        noise_z = np.random.normal(self.mean, self.std, (batch_size, self.latent_dim))
    label_list_z = []
    for i in reversed(range(self.training_days)):
        label_list_z.append(y_train[idx - i])
    multi_labels_z = np.array(label_list_z).reshape(batch_size, self.training_days)
    imgs = X_train[idx]
    gen_imgs = self.generator.predict([noise_z, multi_labels_z])
    print('generated images')


    images1 = FID.scale_images(imgs * 255, (299, 299, 3))
    images2 = FID.scale_images(gen_imgs * 255, (299, 299, 3))
    print('scaled')
    #calculate FID
    images1 = FID.preprocess_input(images1)
    images2 = FID.preprocess_input(images2)
    print('FID preprocessed')
    fid_score = FID.calculate_fid(model, images1, images2)
    self.fid.append(fid_score)
    print('FID (different): %.3f' % fid_score)