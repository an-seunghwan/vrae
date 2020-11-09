#%%
import tensorflow as tf
# import tensorflow_probability as tfp
import tensorflow.keras as K
from tensorflow.keras import layers
from tensorflow.keras import preprocessing
print('TensorFlow version:', tf.__version__)
print('즉시 실행 모드:', tf.executing_eagerly())
print('available GPU:', tf.config.list_physical_devices('GPU'))
from tensorflow.python.client import device_lib
print('==========================================')
print(device_lib.list_local_devices())
tf.debugging.set_log_device_placement(False)
#%%
from tqdm import tqdm
import pandas as pd
import numpy as np
# import random
import math
# import json
import time
import re
import matplotlib.pyplot as plt
from pprint import pprint
from konlpy.tag import Okt
okt = Okt()
import os
os.chdir('/Users/anseunghwan/Documents/uos/generating_text')
print('current directory:', os.getcwd())
from subprocess import check_output
print('=====Data list=====')
print(check_output(["ls", "./data"]).decode("utf8"))
#%%
PARAMS = {
    "batch_size": 1000,
    "data_dim": 784,
    "class_num": 10,
    "latent_dim": 100,
    "epochs": 100, 
    "epsilon_std": 0.01,
    "anneal_rate": 0.0003,
    "init_temperature": 5.0, # shared temperature
    "min_temperature": 0.1,
    "learning_rate": 0.005,
    "stable": False
}

# prior_logits = tf.cast(np.ones((PARAMS['batch_size']*PARAMS['M'], PARAMS['class_num'])) / PARAMS['class_num'], tf.float32)
#%%
class CDVAE(K.models.Model):
    def __init__(self, params):
        super(CDVAE, self).__init__()
        self.params = params
        
        # encoder
        self.enc_dense1 = layers.Dense(512, activation='tanh')
        # self.enc_dense2 = layers.Dense(256, activation='tanh')
        
        # continuous latent
        self.mean = layers.Dense(self.params['latent_dim'], activation='linear')
        self.logvar = layers.Dense(self.params['latent_dim'], activation='tanh')
        
        # discrete latent
        self.logits = layers.Dense(params["class_num"], activation='exponential') # non-negative logits

        # decoder
        self.dec_dense1 = layers.Dense(256, activation='tanh')
        # self.leakyrelu1 = tf.keras.layers.LeakyReLU(alpha=0.2)
        # self.dec_dense2 = layers.Dense(512, activation='linear')
        # self.leakyrelu2 = tf.keras.layers.LeakyReLU(alpha=0.2)
        self.dec_dense3 = layers.Dense(params["data_dim"], activation='sigmoid')
        
    def sample_gumbel(self, shape, eps=1e-20): 
        """Sample from Gumbel(0, 1)"""
        U = tf.random.uniform(shape, minval=0, maxval=1)
        return -tf.math.log(-tf.math.log(U + eps) + eps)

    def gumbel_softmax_sample(self, logits, temperature, stable=True): 
        """
        Draw a sample from the Gumbel-Softmax distribution
        
        logits: [batch_size, n_class] unnormalized log-probs
        temperature: non-negative scalar
        hard: if True, take argmax, but differentiate w.r.t. soft sample y
        """
        eps=1e-20
        if self.params['stable']:
            y_ = (tf.math.log(logits + eps) + self.sample_gumbel(tf.shape(logits))) / temperature
            y = y_ - tf.math.log(tf.reduce_sum(tf.math.exp(y_)) + eps)
            return y
        else:           
            y = tf.math.log(logits + eps) + self.sample_gumbel(tf.shape(logits))
            return tf.nn.softmax(y / temperature)

    def decoder(self, x):
        h = self.dec_dense1(x)
        # h = self.leakyrelu1(h)
        # h = self.dec_dense2(h)
        # h = self.leakyrelu1(h)
        h = self.dec_dense3(h)
        return h

    def call(self, x, tau):
        class_num = self.params["class_num"]   
        latent_dim = self.params["latent_dim"]   

        # encoder
        x = self.enc_dense1(x)
        # x = self.enc_dense2(x)
        
        # continous latent
        mean = self.mean(x)
        logvar = self.logvar(x)
        epsilon = tf.random.normal((latent_dim, ))
        cz = mean + tf.math.exp(logvar / 2) * epsilon 
        
        # discrete latent
        logits = self.logits(x)
        dz = self.gumbel_softmax_sample(logits, tau, self.params['stable'])
        assert dz.shape == (self.params["batch_size"], class_num)
        if self.params['stable']:
            self.sample_dz = tf.math.exp(dz)
            dz = tf.math.exp(dz)
        else:
            self.sample_dz = dz
        
        # decoder
        z = layers.Concatenate()([dz, cz])    
        xhat = self.decoder(z) # (batch_size, 28*28)
        
        return mean, logvar, logits, xhat
#%%
def kl_anneal(step, s, k=0.01):
    return 1 / (1 + math.exp(-k*(step - s)))

# def relaxed_gumbel_softmax_log_density(y, logits, temperature):
#     eps=1e-20
#     # constant is omitted
#     y = tf.reshape(y, [PARAMS['batch_size']*PARAMS['M'], PARAMS['class_num']])
#     gamma = tf.math.log(logits + eps) - temperature*y
#     return (y.shape[-1]-1)*np.math.log(temperature) + \
#                 tf.reduce_sum(gamma, axis=1, keepdims=True) - \
#                     y.shape[-1]*tf.math.log(tf.reduce_sum(tf.math.exp(gamma), axis=1, keepdims=True) + eps)

# def gumbel_loss(y, logits, xhat, x, temperature):
#     # cross-entropy
#     # cross_ent = K.losses.binary_crossentropy(x, xhat, from_logits=False)
#     cross_ent = K.losses.mean_squared_error(x, xhat)
#     cross_ent = tf.reduce_mean(cross_ent, 0, keepdims=True)
    
#     # KL loss
#     kl1 = relaxed_gumbel_softmax_log_density(y, logits, temperature)
#     kl2 = relaxed_gumbel_softmax_log_density(y, prior_logits, temperature)
#     kl_loss = tf.reduce_mean(kl1 - kl2, keepdims=True)
    
#     return cross_ent + kl_loss, cross_ent, kl_loss

def loss_fun(x, xhat, mean, logvar, logits, beta):
    # mse
    mse = K.losses.mean_squared_error(x, xhat)
    mse = tf.reduce_mean(mse, 0, keepdims=True)
    
    # KL loss
    cont_kl = tf.reduce_mean(tf.reduce_sum(0.5 * (tf.math.pow(mean, 2) - 1 + tf.math.exp(logvar) - logvar), axis=-1), keepdims=True)
    logits_ = tf.nn.softmax(logits, axis=-1)
    disc_kl = tf.reduce_mean(tf.reduce_sum(logits_ * (tf.math.log(logits_ + 1e-10) - tf.math.log(1/PARAMS['class_num'])), axis=1), keepdims=True)
    
    return mse + beta * (cont_kl + disc_kl), mse, cont_kl + disc_kl

def get_learning_rate(step, init=PARAMS["learning_rate"]):
    return tf.convert_to_tensor(init * pow(0.95, (step / PARAMS["epochs"])), dtype=tf.float32)
#%%
# data
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
model = CDVAE(PARAMS)
learning_rate = tf.Variable(PARAMS["learning_rate"], trainable=False, name="LR")
optimizer = tf.keras.optimizers.Adam(learning_rate)

# temperature
tau = PARAMS["init_temperature"]
anneal_rate = PARAMS["anneal_rate"]
min_temperature = PARAMS["min_temperature"]
#%%
# Train
for epoch in range(1, PARAMS["epochs"] + 1):
    
    # KL annealing
    beta = kl_anneal(epoch, int(PARAMS["epochs"]/2))
    # beta = kl_anneal(epoch, PARAMS["epochs"])

    for train_x in train_dataset:
        with tf.GradientTape(persistent=True) as tape:
            mean, logvar, logits, xhat  = model(train_x, tau)
            loss, mse, kl_loss = loss_fun(train_x, xhat, mean, logvar, logits, beta)
            
        grad = tape.gradient(loss, model.weights)
        optimizer.apply_gradients(zip(grad, model.weights))
    
    # change temperature and lr
    new_lr = get_learning_rate(epoch)
    learning_rate.assign(new_lr)
    tau = np.maximum(tau * np.exp(-anneal_rate * epoch), min_temperature) # annealing

    print("Epoch:", epoch, ", TRAIN loss:", loss.numpy(), ", Temperature:", tau)
    print("MSE:", mse.numpy(), ", KL loss:", kl_loss.numpy(), ", beta:", beta)
        
    if epoch % 1 == 0:
        losses = []
        for test_x in test_dataset:
            mean, logvar, logits, xhat = model(test_x, tau)
            losses.append(loss_fun(test_x, xhat, mean, logvar, logits, beta)[0].numpy()[0])
        eval_loss = np.mean(losses)
        print("Eval Loss:", eval_loss, "\n")
#%%
np.round(model.sample_dz.numpy()[0], 3)
np.sum(model.sample_dz.numpy()[0])
import matplotlib.pyplot as plt
plt.figure(figsize=(5,5))
plt.imshow(model.sample_dz.numpy()[[0], :])
#%%
def save_plt(original_img, construct_img, code):
    plt.figure(figsize=(5, 10))
    for i in range(0, 15, 3):
        # input img
        plt.subplot(5, 3, i+1)
        plt.imshow(original_img[i, :].reshape(28, 28), cmap='gray')
        plt.axis('off')

        # code
        plt.subplot(5, 3, i+2)
        plt.imshow(code[[i], :], cmap='gray')
        plt.axis('off')

        # output img
        plt.subplot(5, 3, i+3)
        plt.imshow(construct_img[i, :,].reshape((28, 28)), cmap='gray')
        plt.axis('off')
        
    # plt.savefig('./result/vae_rebuilt.png')

mean, logvar, logits, xhat = model(x_train[:PARAMS['batch_size']], tau=2.0)
sample_dz = model.sample_dz.numpy()
xhat = xhat.numpy()
print("sample_y: ", sample_dz.shape, ", logits_x.shape:", xhat.shape)
save_plt(x_train[:PARAMS['batch_size']], xhat, sample_dz)
#%%
'''t-SNE'''
# def save_embed():
C = np.zeros((6000, PARAMS["M"]*PARAMS["class_num"]))
for i in range(0, 6000, 100):
    logits_y, _ = model(x_test[i: i+100], tau=0.5)
    code = tf.reshape(tf.reshape(logits_y, [-1, 30, 10]), [-1, 300])
    # code = tf.reshape(model.sample_y, (100, PARAMS["M"]*PARAMS["N"]))  # (100, 300)
    C[i: i+100] = code

from sklearn.manifold.t_sne import TSNE
tsne = TSNE()
viz = tsne.fit_transform(C)

color = ['aliceblue', 'cyan', 'darkorange', 'fuchsia', 'lightpink', 
            'pink', 'springgreen', 'yellow', 'orange', 'mediumturquoise']
for i in range(0, 6000):
    plt.scatter(viz[i, 0], viz[i, 1], c=color[y_test[i]])
plt.savefig('./result/vae_embed.png')
#%%