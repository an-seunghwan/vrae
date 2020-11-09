#%%
'''
- mode collapsing
- straight through estimator
- fixed tau
- only annealing on KL
- prior means on circle
- Not closed form KL-divergences
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
    "batch_size": 2000,
    "data_dim": 784,
    "class_num": 10,
    "latent_dim": 2,
    "epochs": 100, 
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
class GumbelSoftmaxMixtureVAE(K.models.Model):
    def __init__(self, params):
        super(GumbelSoftmaxMixtureVAE, self).__init__()
        self.params = params
        
        # encoder
        self.enc_dense1 = layers.Dense(256, activation='tanh',
                                      kernel_regularizer=K.regularizers.L1L2(l1=0, l2=0))
        
        # continuous latent
        self.mean_layer = [layers.Dense(self.params['latent_dim'], activation='linear') for _ in range(self.params['class_num'])]
        self.logvar_layer = [layers.Dense(self.params['latent_dim'], activation='linear') for _ in range(self.params['class_num'])]
        
        # discrete latent
        self.logits = layers.Dense(self.params["class_num"], activation='softmax',
                                      kernel_regularizer=K.regularizers.L1L2(l1=0, l2=0)) # non-negative logits

        # decoder
        self.dec_dense1 = layers.Dense(256, activation='tanh',
                                      kernel_regularizer=K.regularizers.L1L2(l1=0, l2=0))
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
    
    def decoder(self, x):
        h = self.dec_dense1(x)
        h = self.dec_dense2(h)
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
        
        # discrete latent
        logits = self.logits(x)
        y = self.gumbel_softmax_sample(logits, tau)
        assert y.shape == (self.params["batch_size"], class_num)
        self.sample_y = y
        
        # mixture sampling
        z_tilde = tf.squeeze(tf.matmul(y[:, tf.newaxis, :], z), axis=1)
        assert z_tilde.shape == (self.params["batch_size"], latent_dim)
                
        # decoder
        xhat = self.decoder(z_tilde) 
        assert xhat.shape == (self.params["batch_size"], self.params['data_dim'])
        
        return mean, logvar, logits, y, z, z_tilde, xhat
#%%
def kl_anneal(epoch, epochs):
    if PARAMS['logistic_anneal']:
        return 1 / (1 + math.exp(-PARAMS['kl_anneal_rate']*(epoch - epochs)))
    else:
        return (1 / epochs) * epoch

def get_learning_rate(epoch, init=PARAMS["learning_rate"]):
    return tf.convert_to_tensor(init * pow(0.95, (epoch / PARAMS["epochs"])), dtype=tf.float32)

def log_gaussian_density(z, mean, logvar):
    return -tf.reduce_sum(logvar, axis=-1) + tf.reduce_sum(tf.divide(tf.pow(z-mean, 2), tf.math.exp(logvar)), axis=-1)

def gumbel_loss(logits, xhat, x, mean, logvar, temperature, beta):
    # reconstruction
    # error = bce_loss(x, xhat)
    error = K.losses.mean_squared_error(x, xhat)
    error = tf.reduce_mean(error, 0, keepdims=True)    
    
    # KL loss by MonteCarlo
    logits_ = logits / tf.tile(tf.reduce_sum(logits, axis=-1, keepdims=True), [1, PARAMS['class_num']])
    log_density1 = - tf.reduce_sum(logvar, axis=-1)/2 - tf.reduce_sum(tf.divide(tf.pow(z-mean, 2), tf.math.exp(logvar)), axis=-1)/2
    log_density2 = - tf.reduce_sum(tf.pow(z-PARAMS['prior_means'], 2), axis=-1)/2
    kl = tf.divide(tf.reduce_sum(logits_ * tf.math.exp(log_density1), axis=-1), tf.reduce_mean(tf.math.exp(log_density2), axis=-1))
    kl_loss = tf.reduce_mean(tf.math.log(kl + 1e-20))
    
    return error + beta * kl_loss, error, kl_loss

def center_reconstruction(model, epoch):
    prior_sample = tf.cast(PARAMS['prior_means'].numpy()[0], tf.float32)
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
        
def example_reconstruction(train_x, y, xhat):
    y = y.numpy()               
    xhat = xhat.numpy()  
    
    plt.figure(figsize=(5, 10))
    for i in range(0, 15, 3):
        # input img
        plt.subplot(5, 3, i+1)
        plt.imshow(train_x[i+5, :].numpy().reshape(28, 28), cmap='gray')
        plt.axis('off')
        # code
        plt.subplot(5, 3, i+2)
        plt.imshow(y[[i+5], :])
        plt.axis('off')
        # output img
        plt.subplot(5, 3, i+3)
        plt.imshow(xhat[i+5, :,].reshape((28, 28)), cmap='gray')
        plt.axis('off')
    if PARAMS['FashionMNIST']:
        plt.savefig('./result/mixturevae_fashionmnist_rebuilt_epoch{}.png'.format(epoch),
                    bbox_inches="tight", pad_inches=0.1)
    else:
        plt.savefig('./result/mixturevae_mnist_rebuilt_epoch{}.png'.format(epoch),
                    bbox_inches="tight", pad_inches=0.1)
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
test_dataset = tf.data.Dataset.from_tensor_slices(x_test).shuffle(test_buffer).batch(batch_size)
#%%
model = GumbelSoftmaxMixtureVAE(PARAMS)
learning_rate = tf.Variable(PARAMS["learning_rate"], trainable=False, name="LR")
optimizer = tf.keras.optimizers.RMSprop(learning_rate)

# tau = PARAMS["init_temperature"]
tau = PARAMS["min_temperature"]
# anneal_rate = PARAMS["temperature_anneal_rate"]
# min_temperature = PARAMS["min_temperature"]
#%%
# Train
for epoch in range(1, PARAMS["epochs"] + 1):
    
    # KL annealing
    beta = kl_anneal(epoch, int(PARAMS["epochs"] / 2))

    for train_x in train_dataset:
        with tf.GradientTape(persistent=True) as tape:
            mean, logvar, logits, y, z, z_tilde, xhat = model(train_x, tau)
            loss, mse, kl_loss = gumbel_loss(logits, xhat, train_x, mean, logvar, tau, beta) # gamma
            
        grad = tape.gradient(loss, model.weights)
        optimizer.apply_gradients(zip(grad, model.weights))
    
    # change temperature and learning rate
    # tau = np.maximum(tau * np.exp(-anneal_rate * epoch), min_temperature) # annealing
    new_lr = get_learning_rate(epoch)
    learning_rate.assign(new_lr)

    print("Epoch:", epoch, ", TRAIN loss:", loss.numpy(), ", Temperature:", tau)
    print("MSE:", mse.numpy(), ", KL loss:", kl_loss.numpy(), ", beta:", beta)
    print(np.round(logits.numpy()[0], 3))
    
    if epoch % 10 == 0:
        center_reconstruction(model, epoch)
        example_reconstruction(train_x, y, xhat)
    
    if epoch % 1 == 0:
        losses = [] # only ELBO loss
        for test_x in test_dataset:
            mean, logvar, logits, y, z, z_tilde, xhat = model(test_x, tau)
            losses.append(gumbel_loss(logits, xhat, test_x, mean, logvar, tau, beta)[0].numpy()[0])
        eval_loss = np.mean(losses)
        print("Eval Loss:", eval_loss, "\n")
#%%
a = np.arange(-12, 12, 2)
b = np.arange(-12, 12, 2)
aa, bb = np.meshgrid(a, b, sparse=True)
grid = []
for b_ in reversed(bb[:, 0]):
    for a_ in aa[0, :]:
        grid.append(np.array([a_, b_]))
grid_output = model.decoder(tf.cast(np.array(grid), tf.float32))
grid_output = grid_output.numpy()
plt.figure(figsize=(20, 20))
for i in range(len(grid)):
    plt.subplot(len(a), len(b), i+1)
    plt.imshow(grid_output[i].reshape(28, 28), cmap='gray')    
    plt.axis('off')
    plt.tight_layout() 
    # plt.savefig('./result/grid_200919.png', 
    #             dpi=600, bbox_inches="tight", pad_inches=0.1)
#%%
zmat = z.numpy().reshape(PARAMS['batch_size']*PARAMS['class_num'], PARAMS['latent_dim'])
zlabel = sum([list(range(PARAMS['class_num']))]*PARAMS['batch_size'], [])
plt.figure(figsize=(15, 15))
plt.scatter(zmat[:, 0], zmat[:, 1], c=zlabel, s=20, cmap=plt.cm.Reds, alpha=1)
# plt.savefig('./result/mixturevae_mnist_z_200921.png')
#%%
'''t-SNE'''
C = np.zeros((10000, PARAMS["latent_dim"]))
for i in range(0, 10000, PARAMS['batch_size']):
    _, _, _, _, _, z_tilde, _ = model(x_test[i: i+PARAMS['batch_size']], tau=1.0)
    C[i: i+PARAMS['batch_size']] = z_tilde
# idx = np.random.choice(10000, 2000, replace=False)
C = C[:, :]
y = y_test[:]
from sklearn.manifold import TSNE
tsne = TSNE()
viz = tsne.fit_transform(C)

color = ['aliceblue', 'cyan', 'darkorange', 'fuchsia', 'lightpink', 
            'pink', 'springgreen', 'yellow', 'orange', 'mediumturquoise']
plt.figure(figsize=(15, 15))
for i in range(0, 10000):
    plt.scatter(viz[i, 0], viz[i, 1], c=color[y[i]])
# if PARAMS['FashionMNIST']:
#     plt.savefig('./result/mixturevae_fashionmnist_tsne_200921.png')
# else:
#     plt.savefig('./result/mixturevae_mnist_tsne_200921.png')
#%%
# r = 6
# prior_means = np.array([[r*np.cos(np.pi/10), r*np.sin(np.pi/10)],
#                         [r*np.cos(3*np.pi/10), r*np.sin(3*np.pi/10)],
#                         [r*np.cos(5*np.pi/10), r*np.sin(5*np.pi/10)],
#                         [r*np.cos(7*np.pi/10), r*np.sin(7*np.pi/10)],
#                         [r*np.cos(9*np.pi/10), r*np.sin(9*np.pi/10)],
#                         [r*np.cos(11*np.pi/10), r*np.sin(11*np.pi/10)],
#                         [r*np.cos(13*np.pi/10), r*np.sin(13*np.pi/10)],
#                         [r*np.cos(15*np.pi/10), r*np.sin(15*np.pi/10)],
#                         [r*np.cos(17*np.pi/10), r*np.sin(17*np.pi/10)],
#                         [r*np.cos(19*np.pi/10), r*np.sin(19*np.pi/10)]])

# plt.scatter(prior_means[:, 0], prior_means[:, 1], c=color, s=100)
# plt.savefig('./result/mixturevae_mnist_prior_means_200921.png')
#%%