import tensorflow as tf
import os
import numpy as np
import time
import matplotlib.pyplot as plt
import glob
import zipfile

from random import random
from argparse import ArgumentParser
from matplotlib import *
from random import random
from tensorflow.keras import layers

#constants

BATCH_SIZE = 10
EPOCHS = 50


def generator_model():
    model_in = layers.Input((720, 1280, 3))
    
    #down
    c1 = layers.Conv2D(16, (3, 3), kernel_initializer = 'he_normal', padding = 'same')(model_in)
    c1 = layers.BatchNormalization()(c1)
    c1 = layers.Activation('relu')(c1)
    c1 = layers.Conv2D(16, (3, 3), kernel_initializer = 'he_normal', padding = 'same')(c1)
    c1 = layers.BatchNormalization()(c1)
    c1 = layers.Activation('relu')(c1)

    p1 = layers.MaxPooling2D((2,2))(c1)
    p1 = layers.Dropout(0.1)(p1)

    c2 = layers.Conv2D(32, (3, 3), kernel_initializer = 'he_normal', padding = 'same')(p1)
    c2 = layers.BatchNormalization()(c2)
    c2 = layers.Activation('relu')(c2)
    c2 = layers.Conv2D(32, (3, 3), kernel_initializer = 'he_normal', padding = 'same')(c2)
    c2 = layers.BatchNormalization()(c2)
    c2 = layers.Activation('relu')(c2)

    p2 = layers.MaxPooling2D((2,2))(c2)
    p2 = layers.Dropout(0.1)(p2)

    c3 = layers.Conv2D(64, (3, 3), kernel_initializer = 'he_normal', padding = 'same')(p2)
    c3 = layers.BatchNormalization()(c3)
    c3 = layers.Activation('relu')(c3)
    c3 = layers.Conv2D(64, (3, 3), kernel_initializer = 'he_normal', padding = 'same')(c3)
    c3 = layers.BatchNormalization()(c3)
    c3 = layers.Activation('relu')(c3)

    p3 = layers.MaxPooling2D((2,2))(c3)
    p3 = layers.Dropout(0.1)(p3)

    c4 = layers.Conv2D(128, (3, 3), kernel_initializer = 'he_normal', padding = 'same')(p3)
    c4 = layers.BatchNormalization()(c4)
    c4 = layers.Activation('relu')(c4)
    c4 = layers.Conv2D(128, (3, 3), kernel_initializer = 'he_normal', padding = 'same')(c4)
    c4 = layers.BatchNormalization()(c4)
    c4 = layers.Activation('relu')(c4)

    p4 = layers.MaxPooling2D((2,2))(c4)
    p4 = layers.Dropout(0.1)(p4)

    c5 = layers.Conv2D(256, (3, 3), kernel_initializer = 'he_normal', padding = 'same')(p4)
    c5 = layers.BatchNormalization()(c5)
    c5 = layers.Activation('relu')(c5)
    c5 = layers.Conv2D(256, (3, 3), kernel_initializer = 'he_normal', padding = 'same')(c5)
    c5 = layers.BatchNormalization()(c5)
    c5 = layers.Activation('relu')(c5)


    noise = tf.random.normal(tf.shape(c5))
    c5 = layers.concatenate([c5, noise])
    
    
    #up
    u6 = layers.Conv2DTranspose(128, (2, 2), strides = (2,2), padding = 'same')(c5)
    u6 = layers.concatenate([u6, c4])
    u6 = layers.Dropout(0.1)(u6)
    c6 = layers.Conv2D(128, (3, 3), kernel_initializer = 'he_normal', padding = 'same')(u6)
    c6 = layers.BatchNormalization()(c6)
    c6 = layers.Activation('relu')(c6)
    c6 = layers.Conv2D(128, (3, 3), kernel_initializer = 'he_normal', padding = 'same')(c6)
    c6 = layers.BatchNormalization()(c6)
    c6 = layers.Activation('relu')(c6) 

    u7 = layers.Conv2DTranspose(64, (2, 2), strides = (2,2), padding = 'same')(c6)
    u7 = layers.concatenate([u7, c3])
    u7 = layers.Dropout(0.1)(u7)
    c7 = layers.Conv2D(64, (3, 3), kernel_initializer = 'he_normal', padding = 'same')(u7)
    c7 = layers.BatchNormalization()(c7)
    c7 = layers.Activation('relu')(c7)
    c7 = layers.Conv2D(64, (3, 3), kernel_initializer = 'he_normal', padding = 'same')(c7)
    c7 = layers.BatchNormalization()(c7)
    c7 = layers.Activation('relu')(c7) 

    u8 = layers.Conv2DTranspose(32, (2, 2), strides = (2,2), padding = 'same')(c7)
    u8 = layers.concatenate([u8, c2])
    u8 = layers.Dropout(0.1)(u8)
    c8 = layers.Conv2D(32, (3, 3), kernel_initializer = 'he_normal', padding = 'same')(u8)
    c8 = layers.BatchNormalization()(c8)
    c8 = layers.Activation('relu')(c8)
    c8 = layers.Conv2D(32, (3, 3), kernel_initializer = 'he_normal', padding = 'same')(c8)
    c8 = layers.BatchNormalization()(c8)
    c8 = layers.Activation('relu')(c8)

    u9 = layers.Conv2DTranspose(16, (2, 2), strides = (2,2), padding = 'same')(c8)
    u9 = layers.concatenate([u9, c1])
    u9 = layers.Dropout(0.1)(u9)
    c9 = layers.Conv2D(16, (3, 3), kernel_initializer = 'he_normal', padding = 'same')(u9)
    c9 = layers.BatchNormalization()(c9)
    c9 = layers.Activation('relu')(c9)
    c9 = layers.Conv2D(16, (3, 3), kernel_initializer = 'he_normal', padding = 'same')(c9)
    c9 = layers.BatchNormalization()(c9)
    c9 = layers.Activation('relu')(c9)

    model_out = layers.Conv2D(3, (1,1), activation = 'sigmoid')(c9)
    model = tf.keras.Model(inputs = [model_in], outputs = [model_out])
    return model
    

def discriminator_model():
    
    input1 = layers.Input((720, 1280, 3))
    input2 = layers.Input((720, 1280, 3))

    c11 = layers.Conv2D(128, (3, 3), strides = (2, 2), padding = 'same', input_shape = [720, 1280, 3])(input1)
    c11 = layers.LeakyReLU()(c11)
    c11 = layers.Dropout(0.15)(c11)
    
    c12 = layers.Conv2D(128, (3, 3), strides = (2, 2), padding = 'same', input_shape = [720, 1280, 3])(input2)
    c12 = layers.LeakyReLU()(c12)
    c12 = layers.Dropout(0.15)(c12)
    
    c21 = layers.Conv2D(64, (3, 3), strides = (2, 2), padding = 'same')(c11)
    c21 = layers.LeakyReLU()(c21)
    c21 = layers.Dropout(0.15)(c21)
    
    c22 = layers.Conv2D(64, (3, 3), strides = (2, 2), padding = 'same')(c12)
    c22 = layers.LeakyReLU()(c22)
    c22 = layers.Dropout(0.15)(c22)
    
    c3 = layers.concatenate([c21, c22])
    c3 = layers.Conv2D(32, (3, 3), strides = (2, 2), padding = 'same')(c3)
    c3 = layers.LeakyReLU()(c3)
    c3 = layers.Dropout(0.15)(c3)
    
    c4 = layers.Conv2D(16, (3, 3), strides = (2, 2), padding = 'same')(c3)
    c4 = layers.LeakyReLU()(c4)
    c4 = layers.Dropout(0.15)(c4)
    
    fla = layers.Flatten()(c4)
    out = layers.Dense(1)(fla)
    
    model = tf.keras.Model(inputs = [input1, input2], outputs = [out])
    return model
    
def normalize(X_train, Y_train, X_test):
    X_train = (X_train - 127.5) / 127.5
    Y_train = (Y_train - 127.5) / 127.5
    X_test = (X_test - 127.5) / 127.5
    return X_train, Y_train, X_test
    
def load_normalize():
    x_train = [image.imread(f) for f in glob.glob('./train_blur/' + "**/*.png", recursive = True)]
    y_train = [image.imread(f) for f in glob.glob('./train_sharp/' + "**/*.png", recursive = True)]
    x_test = [image.imread(f) for f in glob.glob('./test_blur/' + "**/*.png", recursive = True)]
    X_train = np.asarray(x_train)
    Y_train = np.asarray(y_train)
    X_test = np.asarray(x_test)
    return normalize(X_train, Y_train, X_test)
    
def load_data(batch_size):
    X_train, Y_train, X_test = load_normalize()
    train_dataset = tf.data.Dataset.from_tensor_slices((X_train, Y_train)).batch(batch_size)
    return train_dataset

l1_loss = tf.keras.losses.MeanAbsoluteError()
l2_loss = tf.keras.losses.MeanSquaredError()
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits = True)

def discriminator_loss(real_output , generated_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(generated_output), generated_output)
    return real_loss, fake_loss

def generator_loss(fake, sharp, dis_f_loss):
    lam1 = 0.5
    lam2 = 3
    lam3 = 0.5
    return lam1 * l1_loss(fake, sharp) + lam2 * l2_loss(fake, sharp) + lam3 * dis_f_loss

def train():
    for epoch in range(EPOCHS):
        start = time.time()
        
        for (x_batch, y_batch) in train_dataset:
            with tf.GradientTape() as gen, tf.GradientTape() as dis:
                
                gen_images = generator(x_batch, training = True)
                
                real_output = discriminator([x_batch, y_batch], training = True)
                fake_output = discriminator([x_batch, gen_images], training = True)
                
                dis_r_loss, dis_f_loss = discriminator_loss(real_output, fake_output)
                dis_loss = dis_r_loss + dis_f_loss
                gen_loss = generator_loss(gen_images, y_batch, dis_f_loss)
            
            gen_gradients = gen.gradient(gen_loss, generator.trainable_variables)
            dis_gradients = dis.gradient(dis_loss, discriminator.trainable_variables)
            
            generator_optimizer.apply_gradients(zip(gen_gradients, generator.trainable_variables))
            discriminator_optimizer.apply_gradients(zip(dis_gradients, discriminator.trainable_variables))
        
        print("Epoch: {} Time: {}sec".format(epoch + 1, time.time() - start))
        log_array.append([epoch, gen_loss, dis_loss])
        if (epoch + 1) % 3 == 0:
            checkpoint.save(fileprefix = checkpoint_prefix)
        

def load_checkpoint():
    checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))

train_dataset = load_data(BATCH_SIZE)
log_array = []

generator_optimizer = tf.keras.optimizers.Adam()
discriminator_optimizer = tf.keras.optimizers.Adam()
generator = generator_model()
discriminator = discriminator_model()
generator.summary()
discriminator.summary()

checkpoint_dir = './tmp/training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer, 
                                discriminator_optimizer=discriminator_optimizer, 
                                generator=generator,
                                discriminator=discriminator)

#load_checkpoint()
train()
np.save('./tmp/log_array.npy', log_array)