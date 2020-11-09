#%%
'''
- stop overfitting!
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
PARAMS = {
    "batch_size": 1000,
    "data_dim": 784,
    "class_num": 10,
    "latent_dim": 64,
    "epochs": 100, 
    "epsilon_std": 0.01,
    "kl_anneal_rate": 0.1,
    "temperature_anneal_rate": 0.0002,
    "init_temperature": 3.0, # shared temperature
    "min_temperature": 1.0,
    "learning_rate": 0.003,
    "stable": True,
    "FashionMNIST": True
}

# np.random.seed(1)
# prior_means = np.random.uniform(low=-2, high=2, size=(PARAMS['batch_size'], PARAMS['class_num'], PARAMS['latent_dim']))
prior_means = np.zeros((PARAMS['batch_size'], PARAMS['class_num'], PARAMS['latent_dim']))
for i in range(PARAMS['class_num']):
    prior_means[:, i, :] += (i - PARAMS['class_num']/2) * 0.2
prior_means = tf.cast(prior_means, tf.float32)
#%%
class GumbelSoftmaxMixtureVAE(K.models.Model):
    def __init__(self, params):
        super(GumbelSoftmaxMixtureVAE, self).__init__()
        self.params = params
        
        # encoder
        self.enc_dense1 = layers.Dense(256, activation='tanh')
        
        # continuous latent
        self.mean_layer = [layers.Dense(self.params['latent_dim'], activation='linear') for _ in range(self.params['class_num'])]
        self.logvar_layer = [layers.Dense(self.params['latent_dim'], activation='tanh') for _ in range(self.params['class_num'])]
        
        # discrete latent
        self.logits = layers.Dense(self.params["class_num"], activation='exponential') # non-negative logits

        # decoder
        self.dec_dense1 = layers.Dense(256, activation='tanh')
        # self.dec_dense2 = layers.Dense(512, activation='tanh')
        self.dec_dense3 = layers.Dense(self.params["data_dim"], activation='sigmoid')
        
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
        if self.params['stable']:
            y_ = (tf.math.log(logits + eps) + self.sample_gumbel(tf.shape(logits))) / temperature
            y__ = tf.tile(tf.math.log(tf.reduce_sum(tf.math.exp(y_), axis=-1) + eps)[:, tf.newaxis], [1, self.params['class_num']])
            y = y_ - y__
            return y
        else:           
            y = tf.math.log(logits + eps) + self.sample_gumbel(tf.shape(logits))
            return tf.nn.softmax(y / temperature)
    
    def gaussian_sample(self):
        epsilon = tf.random.normal((self.params["class_num"], self.params["latent_dim"]))
        z = self.mean + tf.math.exp(self.logvar / 2) * epsilon 
        return z

    def decoder(self, x):
        h = self.dec_dense1(x)
        h = self.dec_dense3(h)
        return h

    def call(self, x, tau):
        class_num = self.params["class_num"]   
        latent_dim = self.params["latent_dim"]   

        # encoder
        x = self.enc_dense1(x)
        
        # continous latent
        mean = layers.Concatenate(axis=1)([d(x)[:, tf.newaxis, :] for d in self.mean_layer])
        logvar = layers.Concatenate(axis=1)([d(x)[:, tf.newaxis, :] for d in self.logvar_layer])
        epsilon = tf.random.normal((self.params["batch_size"], class_num, latent_dim))
        z = mean + tf.math.exp(logvar / 2) * epsilon 
        assert z.shape == (self.params["batch_size"], class_num, latent_dim)
        self.mean = mean
        self.logvar = logvar
        
        # discrete latent
        logits = self.logits(x)
        y = self.gumbel_softmax_sample(logits, tau)
        assert y.shape == (self.params["batch_size"], class_num)
        if self.params['stable']:
            y = tf.math.exp(y)
            self.sample_y = y
        else:
            self.sample_y = y
            
        z_tilde = tf.squeeze(tf.matmul(y[:, tf.newaxis, :], z), axis=1)
        assert z_tilde.shape == (self.params["batch_size"], latent_dim)
                
        # decoder
        xhat = self.decoder(z_tilde) 
        assert xhat.shape == (self.params["batch_size"], self.params['data_dim'])
        
        return mean, logvar, logits, z, z_tilde, xhat
#%%
def kl_anneal(epoch, s, k=PARAMS['kl_anneal_rate']):
    return 1 / (1 + math.exp(-k*(epoch - s)))

def get_learning_rate(epoch, init=PARAMS["learning_rate"]):
    return tf.convert_to_tensor(init * pow(0.95, (epoch / PARAMS["epochs"])), dtype=tf.float32)

def gaussian_density(z, mean, logvar):
    return tf.math.exp(-(tf.reduce_sum(logvar, axis=-1) + tf.reduce_sum(tf.divide(tf.pow(z-mean, 2), tf.math.exp(logvar)), axis=-1))/2)

def gumbel_loss(logits, xhat, x, mean, logvar, temperature, beta):
    # mse
    mse = K.losses.mean_squared_error(x, xhat)
    mse = tf.reduce_mean(mse, 0, keepdims=True)
    
    # KL loss by closed form
    logits_ = logits / tf.tile(tf.reduce_sum(logits, axis=-1, keepdims=True), [1, PARAMS['class_num']])
    kl1 = tf.reduce_mean(tf.reduce_sum(logits_ * (tf.math.log(logits_ + 1e-20) - tf.math.log(1/PARAMS['class_num'])), axis=1), keepdims=True)
    kl2 = tf.reduce_mean(tf.multiply(tf.reduce_sum(0.5 * (tf.math.pow(mean - prior_means, 2) - 1 + tf.math.exp(logvar) - logvar), axis=-1), logits_))
    kl_loss = kl1 + kl2
    
    return mse + beta * kl_loss, mse, kl_loss
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
test_buffer = len(y_train)

train_dataset = tf.data.Dataset.from_tensor_slices(x_train).shuffle(train_buffer).batch(batch_size)
test_dataset = tf.data.Dataset.from_tensor_slices(x_test).shuffle(test_buffer).batch(batch_size)
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
    beta = kl_anneal(epoch, int(PARAMS["epochs"]/2))

    for train_x in train_dataset:
        with tf.GradientTape(persistent=True) as tape:
            mean, logvar, logits, z, z_tilde, xhat = model(train_x, tau)
            loss, mse, kl_loss = gumbel_loss(logits, xhat, train_x, mean, logvar, tau, beta)
            
        grad = tape.gradient(loss, model.weights)
        optimizer.apply_gradients(zip(grad, model.weights))
    
    # change temperature and learning rate
    tau = np.maximum(tau * np.exp(-anneal_rate * epoch), min_temperature) # annealing
    new_lr = get_learning_rate(epoch)
    learning_rate.assign(new_lr)

    print("Epoch:", epoch, ", TRAIN loss:", loss.numpy(), ", Temperature:", tau)
    print("MSE:", mse.numpy(), ", KL loss:", kl_loss.numpy(), ", beta:", beta)
    
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
        losses = []
        for test_x in test_dataset:
            mean, logvar, logits, z, z_tilde, xhat = model(test_x, tau)
            losses.append(gumbel_loss(logits, xhat, test_x, mean, logvar, tau, beta)[0].numpy()[0])
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