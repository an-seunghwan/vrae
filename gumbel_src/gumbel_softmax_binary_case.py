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
def sample_gumbel(shape, eps=1e-20): 
    '''Sampling from Gumbel(0, 1)'''
    U = tf.random.uniform(shape, minval=0, maxval=1)
    return -tf.math.log(-tf.math.log(U + eps) + eps)

def gumbel_softmax_sample(logits, temperature, stable=True): 
    '''
    Draw a sample from the Gumbel-Softmax distribution
    - logits: unnormalized
    - temperature: non-negative scalar (annealed to 0)
    '''
    eps=1e-20
    y = tf.math.log(logits + eps) + sample_gumbel(tf.shape(logits))
    return tf.nn.softmax(y / temperature)

def sample_logistic(shape, eps=1e-20):
    '''
    sampling from logistic distribution
    = Gumbel - Gumbel
    '''
    U = tf.random.uniform(shape, minval=0, maxval=1)
    return tf.math.log(U + eps) - tf.math.log(1.0 - U + eps)

def sigmoid(x, tau):
    return 1 / (1 + tf.math.exp(-x/tau))

def gumbel_softmax_pdf(x, alpha, tau, eps=1e-20):
    num = tau * alpha * pow(x+eps, -tau-1) * pow(1-x+eps, -tau-1)
    den = pow(alpha * pow(x+eps, -tau) + pow(1-x+eps, -tau), 2)
    return num / den
#%%
logits = np.array([2.0, 0.5]) # non negative value
alpha = logits[0] / logits[1]
temperature = [5.0, 2.0, 1.0, 0.5, 0.1]
x = np.arange(0, 1, 0.01)
s = np.arange(-6, 6, 0.01)

fig, axes = plt.subplots(nrows=5, ncols=2, figsize=(10, 15))
for i, tau in enumerate(temperature):
    L = sample_logistic((1000, ))
    sigmoid_ = sigmoid(s, tau)
    GS = sigmoid(tf.math.log(tf.cast(alpha, tf.float32)) + L, tau)        
    pdf = gumbel_softmax_pdf(x, alpha, tau)
    
    axes[i, 0].hist(L, bins=20, density=True)
    axes[i, 0].plot(s, sigmoid_)
    axes[i, 0].set_title('Logistic samples and sigmoid ({})'.format(tau))
    
    axes[i, 1].hist(GS, bins=20, density=True)
    if i not in [3, 4]:
        axes[i, 1].plot(x, pdf, linewidth=2)
    axes[i, 1].set_title('Histogram of GS and True pdf ({})'.format(tau))

fig.suptitle('')
fig.tight_layout() 
plt.savefig('./result/gs_binary.png', dpi=600)
plt.show()
#%%