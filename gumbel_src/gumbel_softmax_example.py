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
    "batch_size": 100,
    "data_dim": 784,
    "class_num": 10,
    "M": 30,
    "epochs": 50000, 
    "epsilon_std": 0.01,
    "anneal_rate": 0.00003,
    "init_temperature": 1.0,
    "min_temperature": 0.5,
    "learning_rate": 0.001
}
#%%
class GumbelSoftmaxVAE(K.models.Model):
    def __init__(self, params):
        super(GumbelSoftmaxVAE, self).__init__()
        self.params = params
        
        # encoder
        self.enc_dense1 = layers.Dense(512, activation='relu')
        self.enc_dense2 = layers.Dense(256, activation='relu')
        self.enc_dense3 = layers.Dense(params["M"]*params["class_num"])

        # decoder
        self.flatten = layers.Flatten()
        self.dec_dense1 = layers.Dense(256, activation='relu')
        self.dec_dense2 = layers.Dense(512, activation='relu')
        self.dec_dense3 = layers.Dense(params["data_dim"])
        
    def sample_gumbel(self, shape, eps=1e-20): 
        """Sample from Gumbel(0, 1)"""
        U = tf.random.uniform(shape, minval=0, maxval=1)
        return -tf.math.log(-tf.math.log(U + eps) + eps)

    def gumbel_softmax_sample(self, logits, temperature): 
        """ Draw a sample from the Gumbel-Softmax distribution"""
        y = logits + self.sample_gumbel(tf.shape(logits))
        return tf.nn.softmax(y / temperature)

    # def gumbel_softmax(self, logits, temperature, hard=False):
    def gumbel_softmax(self, logits, temperature):
        """
        logits: [batch_size, n_class] unnormalized log-probs
        temperature: non-negative scalar
        hard: if True, take argmax, but differentiate w.r.t. soft sample y
        """
        # y.shape=(batch_size, n_class)
        # row sum = 1, each element = probability
        y = self.gumbel_softmax_sample(logits, temperature)
        # if hard: 
        #     # y를 one-hot vector로 변환
        #     y_hard = tf.cast(tf.equal(y, tf.reduce_max(y, 1, keep_dims=True)), y.dtype)
        #     y = tf.stop_gradient(y_hard - y) + y  # y_hard = y
        return y

    def decoder(self, x):
        # decoder
        h = self.flatten(x)
        h = self.dec_dense1(h)
        h = self.dec_dense2(h)
        h = self.dec_dense3(h)
        return h

    # def call(self, x, tau, hard=False):
    def call(self, x, tau):
        class_num = self.params["class_num"]   # 10
        M = self.params["M"]   # 30

        # encoder
        x = self.enc_dense1(x)
        x = self.enc_dense2(x)
        x = self.enc_dense3(x)   # (batch_size, class_num*M)
        logits_y = tf.reshape(x, [-1, class_num])   # (batch_size*M, class_num)

        # sample
        # y = self.gumbel_softmax(logits_y, tau, hard=hard)
        y = self.gumbel_softmax(logits_y, tau)
        assert y.shape == (self.params["batch_size"]*M, class_num)
        y = tf.reshape(y, [-1, M, class_num])
        self.sample_y = y

        # decoder
        logits_x = self.decoder(y) # (batch_size, 28*28)
        return logits_y, logits_x
#%%
# def kl_anneal(step, s, k=0.001):
#     return 1 / (1 + math.exp(-k*(step - s)))

# def gumbel_loss(model, x, tau, hard):
def gumbel_loss(logits_y, logits_x, x):
    M = PARAMS["M"]   
    class_num = PARAMS["class_num"]   
    
    # cross-entropy
    cross_ent = K.losses.binary_crossentropy(x, logits_x, from_logits=True)
    # cross_ent = K.losses.mean_squared_error(x, logits_x)
    cross_ent = tf.reduce_mean(cross_ent, 0, keepdims=True)
    
    # KL loss
    q_y = tf.nn.softmax(logits_y, axis=-1)   # (batch_size*30, class_num)  
    log_q_y = tf.math.log(q_y + 1e-20)   # (batch_size*30, class_num)  
    kl_tmp = tf.reshape(q_y * (log_q_y - tf.math.log(1.0 / class_num)), [-1, M, class_num])  # (batch_size, class_num, M)
    kl = tf.reduce_sum(kl_tmp, [1, 2])    # shape=(batch_size, 1)
    kl_loss = tf.reduce_mean(kl, keepdims=True)
    
    return cross_ent + kl_loss

# def get_learning_rate(step, init=PARAMS["learning_rate"]):
def get_learning_rate(init=PARAMS["learning_rate"]):
    # return tf.convert_to_tensor(init * pow(0.95, (step / PARAMS["epochs"])), dtype=tf.float32)
    return tf.convert_to_tensor(init * 0.95, dtype=tf.float32)
#%%
# data
(x_train, y_train), (x_test, y_test) = K.datasets.mnist.load_data()
x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))

train_buffer = 60000
batch_size = PARAMS['batch_size']
test_buffer = 10000

train_dataset = tf.data.Dataset.from_tensor_slices(x_train).shuffle(train_buffer).batch(batch_size)
test_dataset = tf.data.Dataset.from_tensor_slices(x_test).shuffle(test_buffer).batch(batch_size)
#%%
model = GumbelSoftmaxVAE(PARAMS)
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
    # beta = kl_anneal(epoch, int(PARAMS["epochs"]/2))

    for train_x in train_dataset:
        with tf.GradientTape() as tape:
            logits_y, logits_x = model(train_x, tau)
            
            # loss = gumbel_loss(model, x, tau, hard)
            loss = gumbel_loss(logits_y, logits_x, train_x)
            
        grad = tape.gradient(loss, model.weights)
        optimizer.apply_gradients(zip(grad, model.weights))
        
        # change temperature and lr
        if epoch % 1000 == 1:
            # temperature annealing
            tau = np.maximum(tau * np.exp(-anneal_rate*epoch), min_temperature)
    
            new_lr = get_learning_rate(epoch)
            learning_rate.assign(new_lr)

    print("Epoch:", epoch, ", TRAIN loss:", loss.numpy(), ", Temperature:", tau)
        
    if epoch % 1 == 0:
        losses = []
        for test_x in test_dataset:
            losses.append(gumbel_loss(logits_y, logits_x, train_x).numpy()[0])
        eval_loss = np.mean(losses)
        print("Eval Loss:", eval_loss, "\n")
#%%
np.round(model.sample_y.numpy()[0], 3)
np.sum(model.sample_y.numpy()[0], axis=1, keepdims=True)
import matplotlib.pyplot as plt
plt.figure(figsize=(15,15))
plt.imshow(model.sample_y.numpy()[0])
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
        plt.imshow(code[i, :, :], cmap='gray')
        plt.axis('off')

        # output img
        plt.subplot(5, 3, i+3)
        plt.imshow(construct_img[i, :,].reshape((28, 28)), cmap='gray')
        plt.axis('off')
        
    plt.savefig('./result/vae_rebuilt.png')

logits_y, logits_x = model(x_train[:PARAMS['batch_size']], tau=0.5)
sample_y = model.sample_y.numpy()                # shape=(100,30,10)
# logits_x = tf.sigmoid(logits_x).numpy()          # shape=(100,784)
logits_x = logits_x.numpy()          # shape=(100,784)
print("sample_y: ", sample_y.shape, ", logits_x.shape:", logits_x.shape)
# code = model.sample_y
save_plt(x_train[:100], logits_x, sample_y)
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