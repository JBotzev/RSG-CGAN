import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

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
        # print('days i ', days[i])
    multi_labels = np.array(label_list).reshape(1, self.training_days)
    gen_imgs = self.generator.predict([noise, multi_labels]).reshape(24)
    # print('gen imgs ', gen_imgs.shape, gen_imgs)
    #
    # print('multi labels ', multi_labels.shape, multi_labels)
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
            axis[j].set_title("Day %d - Wind %d" % (cnt, y_train[days[cnt]]) )
            # axis[i,j].ylim([0,1])
            axis[j].set_xlabel('Hours')
            axis[j].set_ylabel('MW Power Generated')
            cnt += 1
    fig.tight_layout()
    plt.savefig(dir + "Samples %d.png" % epoch)
    self.combined.save(dir + 'model.h5')
    plt.close()

def plot_summary_tests(directory,stats_summ, pvals_summ, title, epoch):
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
