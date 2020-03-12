import tensorflow as tf
import os
import numpy as np
import time
import matplotlib.pyplot as plt
import glob
import zipfile
import PIL
import horovod.tensorflow as hvd

from random import random
from argparse import ArgumentParser
from PIL import Image
from random import random
from tensorflow.keras import layers


hvd.init()

gpus = tf.config.experimental.list_physical_devices('GPU')
print("Gpu's for horovod: ")
print(gpus)

for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
if gpus:
    tf.config.experimental.set_visible_devices(gpus[hvd.local_rank()], 'GPU')
    
#toprintlogs
#tf.debugging.set_log_device_placement(True)


#constants

BATCH_SIZE = 1
EPOCHS = 3

def build_parser():
    parser = ArgumentParser()
    parser.add_argument("-e", "--epochs", type = int, dest = 'epochs', help = "Number of epochs", default = EPOCHS)
    return parser

def hw_flatten(x) :
    return tf.reshape(x, shape=[1, -1 ,x.shape[-1]])
		
def attention(x):
	channels = x.shape[-1]
	f = layers.Conv2D(channels, (1, 1), kernel_initializer = 'he_normal', padding = 'same')(x) 
	g = layers.Conv2D(channels, (1, 1), kernel_initializer = 'he_normal', padding = 'same')(x)
	h = layers.Conv2D(channels, (1, 1), kernel_initializer = 'he_normal', padding = 'same')(x)
	# attention map
	beta = tf.nn.softmax(tf.matmul(hw_flatten(g), hw_flatten(f), transpose_b=True)) 
	o = tf.matmul(beta, hw_flatten(h)) # [bs, N, C]
	gamma = tf.compat.v1.get_variable("gamma", [1], initializer=tf.constant_initializer(0.0))
	o = tf.reshape(o, shape=tf.shape(x)) # [bs, h, w, C]
	final = gamma * o + x
	return final

def generator_model():
    model_in = layers.Input((720, 1280, 3))
    
    #space to depth
    s2d = tf.nn.space_to_depth(model_in, 2, data_format='NHWC')
    
    #down
    c1 = layers.Conv2D(16, (3, 3), kernel_initializer = 'he_normal', padding = 'same')(s2d)
    c1 = layers.BatchNormalization()(c1)
    c1 = layers.LeakyReLU()(c1)
    c1 = layers.Conv2D(16, (3, 3), kernel_initializer = 'he_normal', padding = 'same')(c1)
    c1 = layers.BatchNormalization()(c1)
    c1 = layers.LeakyReLU()(c1)

    p1 = layers.MaxPooling2D((2,2))(c1)
    p1 = layers.Dropout(0.1)(p1)

    c2 = layers.Conv2D(32, (3, 3), kernel_initializer = 'he_normal', padding = 'same')(p1)
    c2 = layers.BatchNormalization()(c2)
    c2 = layers.LeakyReLU()(c2)
    c2 = layers.Conv2D(32, (3, 3), kernel_initializer = 'he_normal', padding = 'same')(c2)
    c2 = layers.BatchNormalization()(c2)
    c2 = layers.LeakyReLU()(c2)

    p2 = layers.MaxPooling2D((2,2))(c2)
    p2 = layers.Dropout(0.1)(p2)

    c3 = layers.Conv2D(64, (3, 3), kernel_initializer = 'he_normal', padding = 'same')(p2)
    c3 = layers.BatchNormalization()(c3)
    c3 = layers.LeakyReLU()(c3)
    c3 = layers.Conv2D(64, (3, 3), kernel_initializer = 'he_normal', padding = 'same')(c3)
    c3 = layers.BatchNormalization()(c3)
    c3 = layers.LeakyReLU()(c3)

    p3 = layers.MaxPooling2D((2,2))(c3)
    p3 = layers.Dropout(0.1)(p3)

    c4 = layers.Conv2D(128, (3, 3), kernel_initializer = 'he_normal', padding = 'same')(p3)
    c4 = layers.BatchNormalization()(c4)
    c4 = layers.LeakyReLU()(c4)
    c4 = layers.Conv2D(128, (3, 3), kernel_initializer = 'he_normal', padding = 'same')(c4)
    c4 = layers.BatchNormalization()(c4)
    c4 = layers.LeakyReLU()(c4)

    p4 = layers.MaxPooling2D((2,2))(c4)
    p4 = layers.Dropout(0.1)(p4)

    c5 = layers.Conv2D(256, (3, 3), kernel_initializer = 'he_normal', padding = 'same')(p4)
    c5 = layers.BatchNormalization()(c5)
    c5 = layers.LeakyReLU()(c5)
    c5 = layers.Conv2D(256, (3, 3), kernel_initializer = 'he_normal', padding = 'same')(c5)
    c5 = layers.BatchNormalization()(c5)
    c5 = layers.LeakyReLU()(c5)

    # Removed noise to check
    
    #Self Attention Part
    c5 = attention(c5)
    noise = tf.random.normal(tf.shape(c5))
    c5 = layers.concatenate([c5, noise])

    #up
    u6 = layers.Conv2DTranspose(128, (2, 2), strides = (2,2), padding = 'same')(c5)
    #padding to match 44 -> 45
    padded = layers.ZeroPadding2D(padding = ((1,0),(0,0)))(u6)
    
    u6 = layers.concatenate([padded, c4])
    u6 = layers.Dropout(0.1)(u6)
    c6 = layers.Conv2D(128, (3, 3), kernel_initializer = 'he_normal', padding = 'same')(u6)
    c6 = layers.BatchNormalization()(c6)
    c6 = layers.LeakyReLU()(c6)
    c6 = layers.Conv2D(128, (3, 3), kernel_initializer = 'he_normal', padding = 'same')(c6)
    c6 = layers.BatchNormalization()(c6)
    c6 = layers.LeakyReLU()(c6) 

    u7 = layers.Conv2DTranspose(64, (2, 2), strides = (2,2), padding = 'same')(c6)
    u7 = layers.concatenate([u7, c3])
    u7 = layers.Dropout(0.1)(u7)
    c7 = layers.Conv2D(64, (3, 3), kernel_initializer = 'he_normal', padding = 'same')(u7)
    c7 = layers.BatchNormalization()(c7)
    c7 = layers.LeakyReLU()(c7)
    c7 = layers.Conv2D(64, (3, 3), kernel_initializer = 'he_normal', padding = 'same')(c7)
    c7 = layers.BatchNormalization()(c7)
    c7 = layers.LeakyReLU()(c7) 

    u8 = layers.Conv2DTranspose(32, (2, 2), strides = (2,2), padding = 'same')(c7)
    u8 = layers.concatenate([u8, c2])
    u8 = layers.Dropout(0.1)(u8)
    c8 = layers.Conv2D(32, (3, 3), kernel_initializer = 'he_normal', padding = 'same')(u8)
    c8 = layers.BatchNormalization()(c8)
    c8 = layers.LeakyReLU()(c8)
    c8 = layers.Conv2D(32, (3, 3), kernel_initializer = 'he_normal', padding = 'same')(c8)
    c8 = layers.BatchNormalization()(c8)
    c8 = layers.LeakyReLU()(c8)

    u9 = layers.Conv2DTranspose(16, (2, 2), strides = (2,2), padding = 'same')(c8)
    u9 = layers.concatenate([u9, c1])
    u9 = layers.Dropout(0.1)(u9)
    c9 = layers.Conv2D(16, (3, 3), kernel_initializer = 'he_normal', padding = 'same')(u9)
    c9 = layers.BatchNormalization()(c9)
    c9 = layers.LeakyReLU()(c9)
    c9 = layers.Conv2D(16, (3, 3), kernel_initializer = 'he_normal', padding = 'same')(c9)
    c9 = layers.BatchNormalization()(c9)
    c9 = layers.LeakyReLU()(c9)
    
    #depthtospace
    d2s = tf.nn.depth_to_space(c9, 2, data_format='NHWC')
    
    #concat and multiscale upsample downsample
    d2s = layers.concatenate([d2s, model_in])
    
    d2s = layers.Conv2D(3, (1,1), kernel_initializer = 'he_normal', padding = 'same')(d2s)
    d2s = layers.BatchNormalization()(d2s)
    d2s = layers.LeakyReLU()(d2s)
    
    #maxpool 2 x 4 x 8 x 16 layers
    max1 = layers.MaxPooling2D((2,2), padding = 'valid')(d2s)
    max2 = layers.MaxPooling2D((4,4), padding = 'valid')(d2s)
    max3 = layers.MaxPooling2D((8,8), padding = 'valid')(d2s)
    max4 = layers.MaxPooling2D((16,16), padding = 'valid')(d2s) 
    
    #upsample
    up1 = layers.UpSampling2D((2,2), interpolation = 'nearest')(max1)
    up2 = layers.UpSampling2D((4,4), interpolation = 'nearest')(max2)
    up3 = layers.UpSampling2D((8,8), interpolation = 'nearest')(max3)
    up4 = layers.UpSampling2D((16,16), interpolation = 'nearest')(max4)
    
    concatpool = layers.concatenate([up1, up2, up3, up4])

    model_out = layers.Conv2DTranspose(3, (1,1), activation = 'tanh')(concatpool)
    model = tf.keras.Model(inputs = [model_in], outputs = [model_out])
    return model
    

def discriminator_model():
    
    input1 = layers.Input((720, 1280, 3))
    input2 = layers.Input((720, 1280, 3))

    c11 = layers.Conv2D(64, (3, 3), strides = (2, 2), padding = 'same', input_shape = [720, 1280, 3])(input1)
    c11 = layers.LeakyReLU()(c11)
    c11 = layers.Dropout(0.15)(c11)
    
    c12 = layers.Conv2D(64, (3, 3), strides = (2, 2), padding = 'same', input_shape = [720, 1280, 3])(input2)
    c12 = layers.LeakyReLU()(c12)
    c12 = layers.Dropout(0.15)(c12)
    
    c21 = layers.Conv2D(16, (3, 3), strides = (2, 2), padding = 'same')(c11)
    c21 = layers.LeakyReLU()(c21)
    c21 = layers.Dropout(0.15)(c21)
    
    c22 = layers.Conv2D(16, (3, 3), strides = (2, 2), padding = 'same')(c12)
    c22 = layers.LeakyReLU()(c22)
    c22 = layers.Dropout(0.15)(c22)
    
    c3 = layers.concatenate([c21, c22])
    c3 = layers.Conv2D(8, (3, 3), strides = (2, 2), padding = 'same')(c3)
    c3 = layers.LeakyReLU()(c3)
    c3 = layers.Dropout(0.15)(c3)
    
    c4 = layers.Conv2D(4, (3, 3), strides = (2, 2), padding = 'same')(c3)
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
    x_train = [f for f in glob.glob('./val_blur/' + "**/*.png", recursive = True)]
    y_train = [f for f in glob.glob('./val_sharp/' + "**/*.png", recursive = True)]
    X_train = np.asarray(x_train)
    Y_train = np.asarray(y_train)
    return X_train, Y_train
    
def load_data(batch_size):
    X_train, Y_train = load_normalize()
    train_dataset = tf.data.Dataset.from_tensor_slices((X_train, Y_train)).repeat().shuffle(3000).batch(batch_size)
    return train_dataset

l1_loss = tf.keras.losses.MeanAbsoluteError()
l2_loss = tf.keras.losses.MeanSquaredError()
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits = True)

def discriminator_loss(real_output , generated_output):
    real_loss = cross_entropy( (np.random.random((real_output.shape))*0.1 + 0.9) * tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy( (np.random.random((generated_output.shape))*0.1) * tf.zeros_like(generated_output), generated_output)
    return real_loss, fake_loss

def generator_loss(fake, sharp, dis_f_loss):
    lam1 = 2
    lam2 = 0.5
    lam3 = 1
    return l1_loss(fake, sharp), l2_loss(fake, sharp), lam1 * l1_loss(fake, sharp) + lam2 * l2_loss(fake, sharp) + lam3 * dis_f_loss

def heatmapwithoutabs(image1, image2):
    image1 = np.asarray(image1)
    image2 = np.asarray(image2)
    image1 = image1[0,:,:,:]
    image2 = image2[0,:,:,:]
    finalimage = np.zeros((image1.shape))
    finalimage[:,:,0] = image1[:,:,0] - image2[:,:,0]
    finalimage[:,:,1] = image1[:,:,1] - image2[:,:,1]
    finalimage[:,:,2] = image1[:,:,2] - image2[:,:,2]
    finalimage = finalimage / 2
    finalimage = tf.convert_to_tensor(finalimage, dtype=tf.float32)
    finalimage = tf.expand_dims(finalimage, 0)
    return finalimage

def train():
    it = 1
    for epoch in range(EPOCHS):
        start = time.time()
        for batch, (x_batch, y_batch) in enumerate(train_dataset.take(3000 // hvd.size())):
            
            x_batch = [np.asarray(Image.open(f)) for f in x_batch.numpy()]
            y_batch = [np.asarray(Image.open(f)) for f in y_batch.numpy()]
            x_batch = [((x - 127.5) / 127.5) for x in x_batch]
            y_batch = [((x - 127.5) / 127.5) for x in y_batch]  
            x_batch = np.asarray(x_batch)
            y_batch = np.asarray(y_batch)
            x_batch = tf.convert_to_tensor(x_batch, dtype=tf.float32)
            y_batch = tf.convert_to_tensor(y_batch, dtype=tf.float32)
            
            with tf.GradientTape() as gen, tf.GradientTape() as dis:
                
                gen_images = generator(x_batch, training = True)
                
                hmapxy = heatmapwithoutabs(x_batch, y_batch)
                real_output = discriminator([x_batch, hmapxy], training = True)
                fake_output = discriminator([x_batch, gen_images], training = True)
                
                dis_r_loss, dis_f_loss = discriminator_loss(real_output, fake_output)
                dis_loss = dis_r_loss + dis_f_loss
                L1, L2, gen_loss = generator_loss(gen_images, hmapxy, dis_f_loss)
                
                inpimg = x_batch.numpy()
                expimg = y_batch.numpy()
                genimg = gen_images.numpy()
                if hvd.rank() == 0:
                    print("gen_imgs max:" , max(genimg))
                genimg = inpimg - 2 * genimg
                
                inpimg = inpimg[0,:,:,:]
                expimg = expimg[0,:,:,:]
                genimg = genimg[0,:,:,:]
                
                inpimg = (inpimg / 2 + 0.5)*255
                expimg = (expimg / 2 + 0.5)*255
                genimg = (genimg / 2 + 0.5)*255
                
                inpimg = tf.convert_to_tensor(inpimg, dtype=tf.uint8)
                expimg = tf.convert_to_tensor(expimg, dtype=tf.uint8)
                genimg = tf.convert_to_tensor(genimg, dtype=tf.uint8)
                
                inpimg = tf.expand_dims(inpimg, 0)
                expimg = tf.expand_dims(expimg, 0)
                genimg = tf.expand_dims(genimg, 0)
               
                psnr_exp = tf.image.psnr(inpimg, expimg, max_val = 255).numpy()[0]
                psnr_res = tf.image.psnr(inpimg, genimg, max_val = 255).numpy()[0]
                
                ssim_exp = tf.image.ssim(inpimg, expimg, max_val = 255).numpy()[0]
                ssim_res = tf.image.ssim(inpimg, genimg, max_val = 255).numpy()[0]
            
            gen_distape = hvd.DistributedGradientTape(gen)
            dis_distape = hvd.DistributedGradientTape(dis)
            gen_gradients = gen_distape.gradient(gen_loss, generator.trainable_variables)
            dis_gradients = dis_distape.gradient(dis_loss, discriminator.trainable_variables)
            

            generator_optimizer.apply_gradients(zip(gen_gradients, generator.trainable_variables))
            discriminator_optimizer.apply_gradients(zip(dis_gradients, discriminator.trainable_variables))
            log_array.append([it, L1, L2, gen_loss, dis_f_loss, dis_loss, psnr_exp, psnr_res, ssim_exp, ssim_res])
            it += 1
        
        if hvd.rank() == 0:
            for i in range(50):
                print("Epoch: {} Time: {}sec".format(epoch + 1, time.time() - start))
        
        if hvd.rank() == 0 and (epoch + 1)%3 == 0:
            checkpoint.save(file_prefix = checkpoint_prefix)
        

def load_checkpoint():
    checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))

def heatmap(image1, image2):
    image1 = (image1 / 2 + 0.5) * 255
    image2 = (image2 / 2 + 0.5) * 255    
    finalimage = np.zeros((image1.shape))
    finalimage[:,:,0] = abs(image1[:,:,0] - image2[:,:,0])
    finalimage[:,:,1] = abs(image1[:,:,1] - image2[:,:,1])
    finalimage[:,:,2] = abs(image1[:,:,2] - image2[:,:,2])
    return finalimage

def test():
    x_name = [f for f in glob.glob('./val_blur/' + "**/*.png", recursive = True)]
    y_name = [f for f in glob.glob('./val_sharp/' + "**/*.png", recursive = True)]
    x_name = x_name[0:20]
    y_name = y_name[0:20]
    x_test = [np.asarray(Image.open(f)) for f in x_name]
    y_test = [np.asarray(Image.open(f)) for f in y_name]
    x_test = [((x - 127.5) / 127.5) for x in x_test]
    y_test = [((x - 127.5) / 127.5) for x in y_test]
    it = 1
    for i in range(20):
        x = x_test[i]
        y = y_test[i]
        inp = tf.convert_to_tensor(x, dtype=tf.float32)
        inp = tf.expand_dims(inp, 0)
        out = generator(inp, training = False)
        out = np.asarray(out)
        out = out[0,:,:,:]
        print(out)
        out = x - 2 * out
        print(out)
        heatmapexp = heatmap(x, y)
        heatmapres = heatmap(x, out)    
        y = Image.fromarray(((y/2 + 0.5)*255).astype(np.uint8))
        y.save('./tmp/'+str(it)+'_exp.png')
        out = Image.fromarray(((out/2 + 0.5)*255).astype(np.uint8))
        out.save('./tmp/'+str(it)+'_res.png')
        x = Image.fromarray(((x/2 + 0.5)*255).astype(np.uint8))
        x.save('./tmp/'+str(it)+'_inp.png')
        heatmapexp = Image.fromarray(heatmapexp.astype(np.uint8))
        heatmapexp.save('./tmp/'+str(it)+'_zexp.png')
        heatmapres = Image.fromarray(heatmapres.astype(np.uint8))
        heatmapres.save('./tmp/'+str(it)+'_zres.png')
        it += 1

parser = build_parser()
options = parser.parse_args()
EPOCHS = options.epochs

print("Epochs are: ", EPOCHS)
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

load_checkpoint()
train()
if hvd.rank == 0:
    test()
suffix = int(random()*10000)
np.save('./tmp/log_array_'+str(suffix)+'.npy', log_array)
