#%%
'''
- mixture t-distribution
- straight through estimator
'''
#%%
import tensorflow as tf
import tensorflow.keras as K
from tensorflow.keras import layers
from tensorflow.keras import preprocessing
print('TensorFlow version:', tf.__version__)
print('Eager Execution Mode:', tf.executing_eagerly())
print('available GPU:', tf.config.list_physical_devices('GPU'))
from tensorflow.python.client import device_lib
print('==========================================')
print(device_lib.list_local_devices())
tf.debugging.set_log_device_placement(False)
#%%
from tqdm import tqdm
import pandas as pd
import numpy as np
import math
import time
import re
import matplotlib.pyplot as plt
from pprint import pprint
import os
os.chdir('/Users/anseunghwan/Documents/uos/generating_text')
#%%
# nrv = tf.random.normal((1, 1000, 2), mean=0.0, stddev=1.0)
# m = tf.reduce_mean(nrv, axis=-1, keepdims=True)
# s = tf.math.sqrt(tf.reduce_sum(tf.math.pow(nrv - m, 2), axis=-1, keepdims=True))
# t = tf.squeeze(tf.cast(np.sqrt(2), tf.float32) * m / s, axis=-1)
# plt.hist(np.sort(t.numpy()[0])[20:-20], density=True)
# np.random.standard_t(1, (10, 5))
# plt.hist(np.sort(np.random.standard_t(1, (1000, )))[20:-20], density=True)
#%%
PARAMS = {
    "batch_size": 2000,
    "data_dim": 784,
    "class_num": 10,
    "latent_dim": 2,
    "df": 3,
    "epochs": 100, 
    "epsilon_std": 0.1,
    "kl_anneal_rate": 0.05,
    "logistic_anneal": True,
    "temperature_anneal_rate": 0.0003,
    "init_temperature": 3.0, # shared temperature
    "min_temperature": 1.0,
    "learning_rate": 0.005,
    "hard": True,
    "FashionMNIST": False
}

# prior_means = np.array([[0.5, 0],
#                         [-0.5, 0],
#                         [1.5, 0],
#                         [1, 1],
#                         [0, 1],
#                         [-1, 1],
#                         [-1.5, 0],
#                         [-1, -1],
#                         [0, -1],
#                         [1, -1]])
# prior_means = prior_means * 1.64

# on the circle
r = 6
prior_means = np.array([[r*np.cos(np.pi/10), r*np.sin(np.pi/10)],
                        [r*np.cos(3*np.pi/10), r*np.sin(3*np.pi/10)],
                        [r*np.cos(5*np.pi/10), r*np.sin(5*np.pi/10)],
                        [r*np.cos(7*np.pi/10), r*np.sin(7*np.pi/10)],
                        [r*np.cos(9*np.pi/10), r*np.sin(9*np.pi/10)],
                        [r*np.cos(11*np.pi/10), r*np.sin(11*np.pi/10)],
                        [r*np.cos(13*np.pi/10), r*np.sin(13*np.pi/10)],
                        [r*np.cos(15*np.pi/10), r*np.sin(15*np.pi/10)],
                        [r*np.cos(17*np.pi/10), r*np.sin(17*np.pi/10)],
                        [r*np.cos(19*np.pi/10), r*np.sin(19*np.pi/10)]])

plt.scatter(prior_means[:, 0], prior_means[:, 1])
prior_means = np.tile(prior_means[np.newaxis, :, :], (PARAMS['batch_size'], 1, 1))
prior_means = tf.cast(prior_means, tf.float32)
PARAMS['prior_means'] = prior_means
#%%
def t_sampling(df):
    return tf.cast(np.random.standard_t(df), tf.float32)

enc_dense1 = layers.Dense(256, activation='tanh',
                                kernel_regularizer=K.regularizers.L1L2(l1=0, l2=0))
x = enc_dense1(train_x)
mean_layer = [layers.Dense(PARAMS['latent_dim'], activation='exponential') for _ in range(PARAMS['class_num'])]
m = layers.Concatenate(axis=1)([d(x)[:, tf.newaxis, :] for d in mean_layer])

logits = layers.Dense(PARAMS["class_num"], activation='exponential',
                                kernel_regularizer=K.regularizers.L1L2(l1=0, l2=0)) # non-negative logits
l = logits(x)
mean = tf.squeeze(tf.matmul(l[:, tf.newaxis, :], m), axis=1)
chi = tf.tile(tf.cast(np.random.chisquare(PARAMS['df'], (PARAMS['batch_size'], 1)), tf.float32), (1, PARAMS['latent_dim']))
z = mean + tf.random.normal(mean.shape) * tf.math.sqrt(PARAMS['df'] / chi)


# decoder
dec_dense1 = layers.Dense(256, activation='tanh',
                                kernel_regularizer=K.regularizers.L1L2(l1=0, l2=0))
dec_dense2 = layers.Dense(PARAMS["data_dim"], activation='sigmoid',
                                kernel_regularizer=K.regularizers.L1L2(l1=0, l2=0))

#%%
class GumbelSoftmaxMixtureVAE(K.models.Model):
    def __init__(self, params):
        super(GumbelSoftmaxMixtureVAE, self).__init__()
        self.params = params
        
        # encoder
        self.enc_dense1 = layers.Dense(256, activation='tanh',
                                      kernel_regularizer=K.regularizers.L1L2(l1=0, l2=0))
        
        # continuous latent
        self.mean_layer = [layers.Dense(self.params['latent_dim'], activation='exponential') for _ in range(self.params['class_num'])]
        
        # discrete latent
        self.logits = layers.Dense(self.params["class_num"], activation='exponential',
                                      kernel_regularizer=K.regularizers.L1L2(l1=0, l2=0)) # non-negative logits

        # decoder
        # self.dec_dense1 = layers.Dense(256, activation='tanh',
        #                               kernel_regularizer=K.regularizers.L1L2(l1=0, l2=0))
        self.dec_dense2 = layers.Dense(self.params["data_dim"], activation='sigmoid',
                                      kernel_regularizer=K.regularizers.L1L2(l1=0, l2=0))
        
    def sample_gumbel(self, shape, eps=1e-20): 
        '''Sampling from Gumbel(0, 1)'''
        U = tf.random.uniform(shape, minval=0, maxval=1)
        return -tf.math.log(-tf.math.log(U + eps) + eps)

    def gumbel_softmax_sample(self, logits, temperature): 
        '''
        Draw a sample from the Gumbel-Softmax distribution
        - logits: unnormalized
        - temperature: non-negative scalar (annealed to 0)
        '''
        eps=1e-20
        y = tf.math.log(logits + eps) + self.sample_gumbel(tf.shape(logits))
        y = tf.nn.softmax(y / temperature)
        if self.params['hard']:
            y_hard = tf.cast(tf.equal(y, tf.math.reduce_max(y, 1, keepdims=True)), y.dtype)
            y = tf.stop_gradient(y_hard - y) + y
        return y
    
    def t_sampling(self, df):
        return tf.cast(np.random.standard_t(df), tf.float32)

    def decoder(self, x):
        # h = self.dec_dense1(x)
        h = self.dec_dense2(x)
        return h

    def call(self, x, tau):
        # encoder
        x = self.enc_dense1(x)
        
        mean = layers.Concatenate(axis=1)([d(x)[:, tf.newaxis, :] for d in self.mean_layer])
        logits = self.logits(x)
        y = self.gumbel_softmax_sample(logits, tau)
        assert y.shape == (self.params["batch_size"], self.params["class_num"])
        self.sample_y = y
        
        m = tf.squeeze(tf.matmul(y[:, tf.newaxis, :], mean), axis=1)
        chi = tf.tile(tf.cast(np.random.chisquare(self.params['df'], (self.params['batch_size'], 1)), tf.float32), (1, self.params['latent_dim']))
        z = m + tf.random.normal((self.params['batch_size'], self.params['latent_dim'])) * tf.math.sqrt(self.params['df'] / chi)
        assert z.shape == (self.params["batch_size"], self.params['latent_dim'])
        
        # decoder
        xhat = self.decoder(z) 
        assert xhat.shape == (self.params["batch_size"], self.params['data_dim'])
        
        return mean, logits, z, xhat
#%%
def kl_anneal(epoch, epochs):
    if PARAMS['logistic_anneal']:
        return PARAMS['gamma'] / (1 + math.exp(-PARAMS['kl_anneal_rate']*(epoch - epochs)))
    else:
        return (PARAMS['gamma'] / epochs) * epoch

def get_learning_rate(epoch, init=PARAMS["learning_rate"]):
    return tf.convert_to_tensor(init * pow(0.95, (epoch / PARAMS["epochs"])), dtype=tf.float32)

# def gaussian_density(z, mean, logvar):
#     return tf.math.exp(-(tf.reduce_sum(logvar, axis=-1) + tf.reduce_sum(tf.divide(tf.pow(z-mean, 2), tf.math.exp(logvar)), axis=-1))/2)

bce = tf.keras.losses.BinaryCrossentropy(from_logits=False)
def gumbel_loss(logits, xhat, x, mean, logvar, temperature, beta):
    # reconstruction
    error = bce(x, xhat)
    
    # KL loss by closed form
    logits_ = logits / tf.tile(tf.reduce_sum(logits, axis=-1, keepdims=True), [1, PARAMS['class_num']])
    kl1 = tf.reduce_mean(tf.reduce_sum(logits_ * (tf.math.log(logits_ + 1e-20) - tf.math.log(1/PARAMS['class_num'])), axis=1), keepdims=True)
    kl2 = tf.reduce_mean(tf.multiply(tf.reduce_sum(0.5 * (tf.math.pow(mean - prior_means, 2) - 1 + tf.math.exp(logvar) - logvar), axis=-1), logits_))
    kl_loss = kl1 + kl2
    
    return error + beta * kl_loss, error, kl_loss
#%%
# data
if PARAMS['FashionMNIST']:
    (x_train, y_train), (x_test, y_test) = K.datasets.fashion_mnist.load_data()
else:
    (x_train, y_train), (x_test, y_test) = K.datasets.mnist.load_data()
x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))

train_buffer = len(x_train)
batch_size = PARAMS['batch_size']
test_buffer = len(x_test)

train_dataset = tf.data.Dataset.from_tensor_slices(x_train).shuffle(train_buffer).batch(batch_size)
# train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(train_buffer).batch(batch_size)
test_dataset = tf.data.Dataset.from_tensor_slices(x_test).shuffle(test_buffer).batch(batch_size)
# test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test)).shuffle(test_buffer).batch(batch_size)
#%%
model = GumbelSoftmaxMixtureVAE(PARAMS)
learning_rate = tf.Variable(PARAMS["learning_rate"], trainable=False, name="LR")
optimizer = tf.keras.optimizers.Adam(learning_rate)

tau = PARAMS["init_temperature"]
anneal_rate = PARAMS["temperature_anneal_rate"]
min_temperature = PARAMS["min_temperature"]
#%%
# Train
for epoch in range(1, PARAMS["epochs"] + 1):
    
    # KL annealing
    beta = kl_anneal(epoch, PARAMS["epochs"]) * PARAMS['gamma']
    # beta = kl_anneal(epoch, PARAMS["epochs"]) 

    for train_x in train_dataset:
        with tf.GradientTape(persistent=True) as tape:
            mean, logits, z, xhat = model(train_x, tau)
            loss, mse, kl_loss = gumbel_loss(logits, xhat, train_x, mean, logvar, tau, beta=0.3) # gamma
            
        grad = tape.gradient(loss, model.weights)
        optimizer.apply_gradients(zip(grad, model.weights))
    
    # change temperature and learning rate
    tau = np.maximum(tau * np.exp(-anneal_rate * epoch), min_temperature) # annealing
    new_lr = get_learning_rate(epoch)
    learning_rate.assign(new_lr)

    print("Epoch:", epoch, ", TRAIN loss:", loss.numpy(), ", Temperature:", tau)
    print("MSE:", mse.numpy(), ", KL loss:", kl_loss.numpy(), ", beta:", 0.3)
    print(np.round(logits.numpy()[0], 3))
    
    if epoch % 10 == 0:
        prior_sample = tf.cast(prior_means.numpy()[0] + np.random.normal(size=(PARAMS['class_num'], PARAMS['latent_dim'])), tf.float32)
        center = model.decoder(prior_sample[tf.newaxis, ...]).numpy()
        plt.figure(figsize=(20, 10))
        for i in range(PARAMS['class_num']):
            plt.subplot(1, PARAMS['class_num'], i+1)
            plt.imshow(center[0, i, :].reshape(28, 28), cmap='gray')
            plt.title(i, fontsize=25)
            plt.axis('off')
            plt.tight_layout() 
        if PARAMS['FashionMNIST']:
            plt.savefig('./result/mixturevae_fashionmnist_center_epoch{}.png'.format(epoch),
                        bbox_inches="tight", pad_inches=0.1)
        else:
            plt.savefig('./result/mixturevae_mnist_center_epoch{}.png'.format(epoch),
                        bbox_inches="tight", pad_inches=0.1)
            
        sample_y = model.sample_y.numpy()               
        xhat = xhat.numpy()  
        
        plt.figure(figsize=(5, 10))
        for i in range(0, 15, 3):
            # input img
            plt.subplot(5, 3, i+1)
            plt.imshow(train_x[i+5, :].numpy().reshape(28, 28), cmap='gray')
            plt.axis('off')

            # code
            plt.subplot(5, 3, i+2)
            plt.imshow(sample_y[[i+5], :])
            plt.axis('off')

            # output img
            plt.subplot(5, 3, i+3)
            plt.imshow(xhat[i+5, :,].reshape((28, 28)), cmap='gray')
            plt.axis('off')
        if PARAMS['FashionMNIST']:
            plt.savefig('./result/mixturevae_fashionmnist_rebuilt_epoch{}.png'.format(epoch),
                        bbox_inches="tight", pad_inches=0.1)
        else:
            plt.savefig('./result/mixturevae_mnist_tsne_epoch{}.png'.format(epoch),
                        bbox_inches="tight", pad_inches=0.1)
    
    if epoch % 1 == 0:
        losses = [] # only ELBO loss
        for test_x in test_dataset:
            mean, logvar, logits, z, z_tilde, xhat = model(test_x, tau)
            losses.append(gumbel_loss(logits, xhat, test_x, mean, logvar, tau, beta=0.3)[0].numpy()[0])
        eval_loss = np.mean(losses)
        print("Eval Loss:", eval_loss, "\n")
#%%
# def save_plt(original_img, construct_img, code):
#     plt.figure(figsize=(5, 10))
#     for i in range(0, 15, 3):
#         # input img
#         plt.subplot(5, 3, i+1)
#         plt.imshow(original_img[i+5, :].reshape(28, 28), cmap='gray')
#         plt.axis('off')

#         # code
#         plt.subplot(5, 3, i+2)
#         plt.imshow(code[[i+5], :])
#         plt.axis('off')

#         # output img
#         plt.subplot(5, 3, i+3)
#         plt.imshow(construct_img[i+5, :,].reshape((28, 28)), cmap='gray')
#         plt.axis('off')
#     if PARAMS['FashionMNIST']:
#         plt.savefig('./result/mixturevae_fashionmnist_rebuilt_200914.png')
#     else:
#         plt.savefig('./result/mixturevae_mnist_tsne_200914.png')

# mean, logvar, logits, z, z_tilde, xhat= model(x_train[:PARAMS['batch_size']], tau=1.0)
# sample_y = model.sample_y.numpy()               
# np.sum(sample_y, axis=1)
# xhat = xhat.numpy()          
# print("sample_y: ", sample_y.shape, ", xhat.shape:", xhat.shape)
# save_plt(x_train[:PARAMS['batch_size']], xhat, sample_y)
 #%%
# prior_sample = tf.cast(prior_means.numpy()[0] + np.random.normal(size=(PARAMS['class_num'], PARAMS['latent_dim'])), tf.float32)
# center = model.decoder(prior_sample[tf.newaxis, ...]).numpy()
# plt.figure(figsize=(20, 10))
# for i in range(PARAMS['class_num']):
#     plt.subplot(1, PARAMS['class_num'], i+1)
#     plt.imshow(center[0, i, :].reshape(28, 28), cmap='gray')
#     plt.title(i, fontsize=25)
#     plt.axis('off')
#     plt.tight_layout() 
# if PARAMS['FashionMNIST']:
#     plt.savefig('./result/mixturevae_fashionmnist_center_200914.png')
# else:
#     plt.savefig('./result/mixturevae_mnist_center_200914.png')
#%%
'''t-SNE'''
C = np.zeros((10000, PARAMS["latent_dim"]))
for i in range(0, 10000, PARAMS['batch_size']):
    _, _, _, _, z_tilde, _ = model(x_test[i: i+PARAMS['batch_size']], tau=1.0)
    C[i: i+PARAMS['batch_size']] = z_tilde
idx = np.random.choice(10000, 1000, replace=False)
C = C[idx, :]
y = y_test[idx]
from sklearn.manifold import TSNE
tsne = TSNE()
viz = tsne.fit_transform(C)

color = ['aliceblue', 'cyan', 'darkorange', 'fuchsia', 'lightpink', 
            'pink', 'springgreen', 'yellow', 'orange', 'mediumturquoise']
for i in range(0, 1000):
    plt.scatter(viz[i, 0], viz[i, 1], c=color[y[i]])
if PARAMS['FashionMNIST']:
    plt.savefig('./result/mixturevae_fashionmnist_tsne_200914.png')
else:
    plt.savefig('./result/mixturevae_mnist_tsne_200914.png')
#%%