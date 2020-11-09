#%%
# '''stable setting!'''
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
    "latent_dim": 64,
    "epochs": 100, 
    "epsilon_std": 0.01,
    "kl_anneal_rate": 0.2,
    "temp_anneal_rate": 0.0002,
    "init_temperature": 4.5, # shared temperature
    "min_temperature": 1.5,
    "learning_rate": 0.0035,
    "stable": False
}

# prior_logits = tf.cast(np.ones((PARAMS['batch_size']*PARAMS['M'], PARAMS['class_num'])) / PARAMS['class_num'], tf.float32)
np.random.seed(1)
# prior_means = np.random.uniform(low=-2, high=2, size=(PARAMS['batch_size'], PARAMS['class_num'], PARAMS['latent_dim']))
prior_means = np.zeros((PARAMS['batch_size'], PARAMS['class_num'], PARAMS['latent_dim']))
for i in range(PARAMS['class_num']):
    prior_means[:, i, :] += (i - PARAMS['class_num']/2) * 0.2
prior_means = tf.cast(prior_means, tf.float32)
#%%
class MixtureVAE(K.models.Model):
    def __init__(self, params):
        super(MixtureVAE, self).__init__()
        self.params = params
        self.prior_means = prior_means
        
        # encoder
        self.enc_dense1 = layers.Conv2D(filters=32, kernel_size=3, strides=(2, 2), activation='relu')
        self.enc_dense2 = layers.Conv2D(filters=64, kernel_size=3, strides=(2, 2), activation='relu')
        self.enc_dense3 = layers.Dense(128, activation='tanh')
        
        # continuous latent
        self.mean = [layers.Dense(self.params['latent_dim'], activation='linear') for _ in range(self.params['class_num'])]
        self.logvar = [layers.Dense(self.params['latent_dim'], activation='tanh') for _ in range(self.params['class_num'])]
        
        # discrete latent
        self.logits = layers.Dense(self.params["class_num"], activation='exponential') # non-negative logits

        # decoder
        # self.dec_dense1 = layers.Dense(256, activation='tanh')
        # self.dec_dense2 = layers.Dense(512, activation='tanh')
        # self.dec_dense3 = layers.Dense(self.params["data_dim"], activation='sigmoid')
        
        self.dec_dense1 = layers.Dense(units=7*7*32, activation='relu')
        self.dec_dense2 = layers.Conv2DTranspose(filters=64, kernel_size=3, strides=2, padding='same', activation='relu')
        self.dec_dense3 = layers.Conv2DTranspose(filters=32, kernel_size=3, strides=2, padding='same', activation='relu')
        self.dec_dense4 = layers.Conv2DTranspose(filters=1, kernel_size=3, strides=1, padding='same')
        
    def sample_gumbel(self, shape, eps=1e-20): 
        """
        Sample from Gumbel(0, 1)
        """
        U = tf.random.uniform(shape, minval=0, maxval=1)
        return -tf.math.log(-tf.math.log(U + eps) + eps)

    def gumbel_softmax_sample(self, logits, temperature): 
        """
        Draw a sample from the Gumbel-Softmax distribution
        logits: [batch_size, n_class] unnormalized log-probs
        temperature: non-negative scalar
        """
        eps=1e-20
        y = tf.math.log(logits + eps) + self.sample_gumbel(tf.shape(logits))
        return tf.nn.softmax(y / temperature)

    def decoder(self, z):
        h = self.dec_dense1(z)
        h = tf.reshape(h, [-1, 7, 7, 32])
        h = self.dec_dense2(h)
        h = self.dec_dense3(h)
        h = self.dec_dense4(h)
        
        return h
    
    def call(self, x, tau):
        class_num = self.params["class_num"]   
        latent_dim = self.params["latent_dim"]   

        # encoder
        x = self.enc_dense1(x)
        x = self.enc_dense2(x)
        x = self.enc_dense3(layers.Flatten()(x))
        
        # latent
        mean = layers.Concatenate(axis=1)([d(x)[:, tf.newaxis, :] for d in self.mean])
        logvar = layers.Concatenate(axis=1)([d(x)[:, tf.newaxis, :] for d in self.logvar])
        epsilon = tf.random.normal((class_num, latent_dim)) 
        z = mean + tf.math.exp(logvar / 2) * epsilon
        
        logits = self.logits(x)
        pi = self.gumbel_softmax_sample(logits, tau)
        assert pi.shape == (self.params["batch_size"], class_num)
        self.sample_pi = pi
        
        z_tilde = tf.squeeze(tf.matmul(pi[:, tf.newaxis, :], z), axis=1)
        
        # decoder
        xhat = self.decoder(z_tilde) # (batch_size, 28*28)
        
        return mean, logvar, logits, z_tilde, pi, xhat
#%%
# data
(x_train, y_train), (x_test, y_test) = K.datasets.fashion_mnist.load_data()
x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
# x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
# x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))
x_train = x_train[..., tf.newaxis]
y_train = y_train[..., tf.newaxis]
x_test = x_test[..., tf.newaxis]
y_test = y_test[..., tf.newaxis]

train_buffer = len(x_train)
batch_size = PARAMS['batch_size']
test_buffer = len(y_train)

x_train_dataset = tf.data.Dataset.from_tensor_slices(x_train).shuffle(train_buffer).batch(batch_size)
x_test_dataset = tf.data.Dataset.from_tensor_slices(x_test).shuffle(test_buffer).batch(batch_size)
y_train_dataset = tf.data.Dataset.from_tensor_slices(y_train).shuffle(train_buffer).batch(batch_size)
y_test_dataset = tf.data.Dataset.from_tensor_slices(y_test).shuffle(test_buffer).batch(batch_size)
#%%
def kl_anneal(step, s):
    return 1 / (1 + math.exp(-PARAMS['kl_anneal_rate']*(step - s)))

def loss_fun(x, xhat, y, mean, logvar, logits, beta):
    # mse
    mse = K.losses.mean_squared_error(tf.reshape(x, [-1, PARAMS['data_dim']]), tf.reshape(xhat, [-1, PARAMS['data_dim']]))
    mse = tf.reduce_mean(mse, axis=0, keepdims=True)
    
    logits_ = logits / tf.reduce_sum(logits, axis=-1, keepdims=True)
    
    # cross-entropy
    cross_ent = tf.reduce_mean(K.losses.sparse_categorical_crossentropy(y, logits_, from_logits=False))
    
    # KL loss
    kl1 = tf.reduce_mean(tf.reduce_sum(logits_ * (tf.math.log(logits_ + 1e-20) - tf.math.log(1/PARAMS['class_num'])), axis=1), keepdims=True)
    kl2 = tf.reduce_mean(tf.multiply(tf.reduce_sum(0.5 * (tf.math.pow(mean - prior_means, 2) - 1 + tf.math.exp(logvar) - logvar), axis=-1), logits_))
    
    return mse + 0.1*cross_ent + beta * (kl1 + kl2), mse + 0.1*cross_ent, kl1 + kl2

def get_learning_rate(step, init=PARAMS["learning_rate"]):
    return tf.convert_to_tensor(init * pow(0.95, (step / PARAMS["epochs"])), dtype=tf.float32)
#%%
model = MixtureVAE(PARAMS)
learning_rate = tf.Variable(PARAMS["learning_rate"], trainable=False, name="LR")
optimizer = tf.keras.optimizers.Adam(learning_rate)

# temperature
tau = PARAMS["init_temperature"]
temp_anneal_rate = PARAMS["temp_anneal_rate"]
min_temperature = PARAMS["min_temperature"]
#%%
# Train
for epoch in range(1, PARAMS["epochs"] + 1):
    
    # KL annealing
    beta = kl_anneal(epoch, int(PARAMS["epochs"]/2))
    # beta = kl_anneal(epoch, PARAMS["epochs"])

    for train_x, train_y in zip(x_train_dataset, y_train_dataset):
        with tf.GradientTape(persistent=True) as tape:
            mean, logvar, logits, z, pi, xhat  = model(train_x, tau)
            loss, mse, kl_loss = loss_fun(train_x, xhat, train_y, mean, logvar, logits, beta)
            
        grad = tape.gradient(loss, model.weights)
        optimizer.apply_gradients(zip(grad, model.weights))
    
    # change temperature and lr
    new_lr = get_learning_rate(epoch)
    learning_rate.assign(new_lr)
    tau = np.maximum(tau * np.exp(-temp_anneal_rate * epoch), min_temperature) # annealing

    print("Epoch:", epoch, ", TRAIN loss:", loss.numpy(), ", Temperature:", tau)
    print("MSE+CrossEnt:", mse.numpy(), ", KL loss:", kl_loss.numpy(), ", beta:", beta)
        
    if epoch % 1 == 0:
        losses = []
        for test_x, test_y in zip(x_test_dataset, y_test_dataset): 
            mean, logvar, logits, z, pi, xhat = model(test_x, tau)
            losses.append(loss_fun(test_x, xhat, test_y, mean, logvar, logits, beta)[0].numpy()[0])
        eval_loss = np.mean(losses)
        print("Eval Loss:", eval_loss, "\n")
        for i in range(5):
            # true
            plt.subplot(2, 5, i+1)
            plt.imshow(test_x.numpy()[..., 0][i].reshape(28, 28), cmap='gray')
            plt.axis('off')
            # prediction
            plt.subplot(2, 5, i+6)
            plt.imshow(xhat.numpy()[..., 0][i].reshape(28, 28), cmap='gray')
            plt.axis('off')
        plt.show()
        if eval_loss < 0.03:
            break
#%%
np.round(model.sample_pi.numpy()[0], 3)
np.sum(model.sample_pi.numpy()[0])
import matplotlib.pyplot as plt
plt.figure(figsize=(5,5))
plt.imshow(model.sample_pi.numpy()[[0], :])
#%%
def save_plt(original_img, construct_img, code):
    plt.figure(figsize=(7, 14))
    for i in range(0, 30, 3):
        # input img
        plt.subplot(10, 3, i+1)
        plt.imshow(original_img[i, :].reshape(28, 28), cmap='gray')
        plt.axis('off')

        # code
        plt.subplot(10, 3, i+2)
        plt.imshow(code[[i], :], cmap='gray')
        plt.axis('off')

        # output img
        plt.subplot(10, 3, i+3)
        plt.imshow(construct_img[i, :,].reshape((28, 28)), cmap='gray')
        plt.axis('off')
        
    # plt.savefig('./result/vae_fashion_rebuilt_200903.png')

mean, logvar, logits, z, pi, xhat = model(x_train[:PARAMS['batch_size']], tau)
sample_pi = model.sample_pi.numpy()
xhat = xhat.numpy()
print("sample_y: ", sample_pi.shape, ", logits_x.shape:", xhat.shape)
save_plt(x_train[:PARAMS['batch_size']], xhat, sample_pi)
#%%
'''t-SNE'''
# def save_embed():
C = np.zeros((10000, PARAMS['latent_dim']))
for i in range(0, 10000, PARAMS['batch_size']):
    _, _, _, z, _, _ = model(x_test[i: i+PARAMS['batch_size']], tau)
    C[i: i+PARAMS['batch_size']] = z
idx = np.random.choice(10000, 1000, replace=False)
C = C[idx, :]
y = y_test[idx]
from sklearn.manifold import TSNE
tsne = TSNE()
viz = tsne.fit_transform(C)

color = ['aliceblue', 'cyan', 'darkorange', 'fuchsia', 'lightpink', 
            'pink', 'springgreen', 'yellow', 'orange', 'mediumturquoise']
for i in range(0, 1000):
    plt.scatter(viz[i, 0], viz[i, 1], c=color[y[i, :][0]])
# plt.savefig('./result/vae_fashion_embed_200903.png')
#%%
pm = tf.cast(prior_means.numpy()[[0]], tf.float32)
center_x = model.decoder(pm).numpy()[..., 0]
plt.figure(figsize=(15, 15))
for i in range(PARAMS['class_num']):
    plt.subplot(10, 1, i+1)
    plt.imshow(center_x[i].reshape(28, 28), cmap='gray')
    plt.axis('off')
# plt.savefig('./result/vae_fashion_center_200903.png')
#%%