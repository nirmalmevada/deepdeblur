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
Lambda1 = 1
Lambda2 = 1
Lambda3 = 1

def build_parser():
    parser = ArgumentParser()
    parser.add_argument("-e", "--epochs", type = int, dest = 'epochs', help = "Number of epochs", default = EPOCHS)
    parser.add_argument("-l1", type = int, dest = 'lambda1', help = "L1 loss factor", default = Lambda1)
    parser.add_argument("-l2", type = int, dest = 'lambda2', help = "L2 loss factor", default = Lambda2)
    parser.add_argument("-l3", type = int, dest = 'lambda3', help = "L3 loss factor", default = Lambda3)
    return parser

class DeformableConvLayer(layers.Conv2D):
    """Only support "channel last" data format"""
    def __init__(self,
                 filters,
                 kernel_size,
                 strides=(1, 1),
                 padding='valid',
                 data_format=None,
                 dilation_rate=(1, 1),
                 num_deformable_group=None,
                 activation=None,
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 **kwargs):
        """`kernel_size`, `strides` and `dilation_rate` must have the same value in both axis.
        :param num_deformable_group: split output channels into groups, offset shared in each group. If
        this parameter is None, then set  num_deformable_group=filters.
        """
        super().__init__(
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            data_format=data_format,
            dilation_rate=dilation_rate,
            activation=activation,
            use_bias=use_bias,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer,
            kernel_constraint=kernel_constraint,
            bias_constraint=bias_constraint,
            **kwargs)
        self.kernel = None
        self.bias = None
        self.offset_layer_kernel = None
        self.offset_layer_bias = None
        if num_deformable_group is None:
            num_deformable_group = filters
        if filters % num_deformable_group != 0:
            raise ValueError('"filters" mod "num_deformable_group" must be zero')
        self.num_deformable_group = num_deformable_group

    def build(self, input_shape):
        input_dim = int(input_shape[-1])
        # kernel_shape = self.kernel_size + (input_dim, self.filters)
        # we want to use depth-wise conv
        kernel_shape = self.kernel_size + (self.filters * input_dim, 1)
        self.kernel = self.add_weight(
            name='kernel',
            shape=kernel_shape,
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
            trainable=True,
            dtype=self.dtype)
        if self.use_bias:
            self.bias = self.add_weight(
                name='bias',
                shape=(self.filters,),
                initializer=self.bias_initializer,
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint,
                trainable=True,
                dtype=self.dtype)

        # create offset conv layer
        offset_num = self.kernel_size[0] * self.kernel_size[1] * self.num_deformable_group
        self.offset_layer_kernel = self.add_weight(
            name='offset_layer_kernel',
            shape=self.kernel_size + (input_dim, offset_num * 2),  # 2 means x and y axis
            initializer=tf.zeros_initializer(),
            regularizer=self.kernel_regularizer,
            trainable=True,
            dtype=self.dtype)
        self.offset_layer_bias = self.add_weight(
            name='offset_layer_bias',
            shape=(offset_num * 2,),
            initializer=tf.zeros_initializer(),
            # initializer=tf.random_uniform_initializer(-5, 5),
            regularizer=self.bias_regularizer,
            trainable=True,
            dtype=self.dtype)
        self.built = True

    def call(self, inputs, training=None, **kwargs):
        # get offset, shape [batch_size, out_h, out_w, filter_h, * filter_w * channel_out * 2]
        offset = tf.nn.conv2d(inputs,
                              filters=self.offset_layer_kernel,
                              strides=[1, *self.strides, 1],
                              padding=self.padding.upper(),
                              dilations=[1, *self.dilation_rate, 1])
        offset += self.offset_layer_bias

        # add padding if needed
        inputs = self._pad_input(inputs)

        # some length
        batch_size = tf.shape(inputs)[0]
        channel_in = tf.keras.backend.int_shape(inputs)[-1]
        # batch_size = int(inputs.get_shape()[0])
        # channel_in = int(inputs.get_shape()[-1])
        in_h, in_w = [int(i) for i in inputs.get_shape()[1: 3]]  # input feature map size
        out_h, out_w = [int(i) for i in offset.get_shape()[1: 3]]  # output feature map size
        filter_h, filter_w = self.kernel_size

        # get x, y axis offset
        offset = tf.reshape(offset, [batch_size, out_h, out_w, -1, 2])
        y_off, x_off = offset[:, :, :, :, 0], offset[:, :, :, :, 1]

        # input feature map gird coordinates
        y, x = self._get_conv_indices([in_h, in_w])
        y, x = [tf.expand_dims(i, axis=-1) for i in [y, x]]
        y, x = [tf.tile(i, [batch_size, 1, 1, 1, self.num_deformable_group]) for i in [y, x]]
        # y, x = [tf.reshape(i, [*i.shape[0: 3], -1]) for i in [y, x]]
        y, x = [tf.reshape(i, (tf.shape(i)[0], tf.shape(i)[1], tf.shape(i)[2], -1)) for i in [y, x]]
        y, x = [tf.cast(i, 'float32') for i in [y, x]]

        # add offset
        y, x = y + y_off, x + x_off
        y = tf.clip_by_value(y, 0, in_h - 1)
        x = tf.clip_by_value(x, 0, in_w - 1)

        # get four coordinates of points around (x, y)
        y0, x0 = [tf.cast(tf.math.floor(i), 'int32') for i in [y, x]]
        y1, x1 = y0 + 1, x0 + 1
        # clip
        y0, y1 = [tf.clip_by_value(i, 0, in_h - 1) for i in [y0, y1]]
        x0, x1 = [tf.clip_by_value(i, 0, in_w - 1) for i in [x0, x1]]

        # get pixel values
        indices = [[y0, x0], [y0, x1], [y1, x0], [y1, x1]]
        p0, p1, p2, p3 = [DeformableConvLayer._get_pixel_values_at_point(inputs, i) for i in indices]

        # cast to float
        x0, x1, y0, y1 = [tf.cast(i, 'float32') for i in [x0, x1, y0, y1]]
        # weights
        w0 = (y1 - y) * (x1 - x)
        w1 = (y1 - y) * (x - x0)
        w2 = (y - y0) * (x1 - x)
        w3 = (y - y0) * (x - x0)
        # expand dim for broadcast
        w0, w1, w2, w3 = [tf.expand_dims(i, axis=-1) for i in [w0, w1, w2, w3]]
        # bilinear interpolation
        pixels = tf.add_n([w0 * p0, w1 * p1, w2 * p2, w3 * p3])

        # reshape the "big" feature map
        pixels = tf.reshape(pixels, [batch_size, out_h, out_w, filter_h, filter_w, self.num_deformable_group, channel_in])
        pixels = tf.transpose(pixels, [0, 1, 3, 2, 4, 5, 6])
        pixels = tf.reshape(pixels, [batch_size, out_h * filter_h, out_w * filter_w, self.num_deformable_group, channel_in])

        # copy channels to same group
        feat_in_group = self.filters // self.num_deformable_group
        pixels = tf.tile(pixels, [1, 1, 1, 1, feat_in_group])
        pixels = tf.reshape(pixels, [batch_size, out_h * filter_h, out_w * filter_w, -1])

        # depth-wise conv
        out = tf.nn.depthwise_conv2d(pixels, self.kernel, [1, filter_h, filter_w, 1], 'VALID')
        # add the output feature maps in the same group
        out = tf.reshape(out, [batch_size, out_h, out_w, self.filters, channel_in])
        out = tf.reduce_sum(out, axis=-1)
        if self.use_bias:
            out += self.bias
        return self.activation(out)

    def _pad_input(self, inputs):
        """Check if input feature map needs padding, because we don't use the standard Conv() function.
        :param inputs:
        :return: padded input feature map
        """
        # When padding is 'same', we should pad the feature map.
        # if padding == 'same', output size should be `ceil(input / stride)`
        if self.padding == 'same':
            in_shape = inputs.get_shape().as_list()[1: 3]
            padding_list = []
            for i in range(2):
                filter_size = self.kernel_size[i]
                dilation = self.dilation_rate[i]
                dilated_filter_size = filter_size + (filter_size - 1) * (dilation - 1)
                same_output = (in_shape[i] + self.strides[i] - 1) // self.strides[i]
                valid_output = (in_shape[i] - dilated_filter_size + self.strides[i]) // self.strides[i]
                if same_output == valid_output:
                    padding_list += [0, 0]
                else:
                    p = dilated_filter_size - 1
                    p_0 = p // 2
                    padding_list += [p_0, p - p_0]
            if sum(padding_list) != 0:
                padding = [[0, 0],
                           [padding_list[0], padding_list[1]],  # top, bottom padding
                           [padding_list[2], padding_list[3]],  # left, right padding
                           [0, 0]]
                inputs = tf.pad(inputs, padding)
        return inputs

    def _get_conv_indices(self, feature_map_size):
        """the x, y coordinates in the window when a filter sliding on the feature map
        :param feature_map_size:
        :return: y, x with shape [1, out_h, out_w, filter_h * filter_w]
        """
        feat_h, feat_w = [int(i) for i in feature_map_size[0: 2]]

        x, y = tf.meshgrid(tf.range(feat_w), tf.range(feat_h))
        x, y = [tf.reshape(i, [1, *i.get_shape(), 1]) for i in [x, y]]  # shape [1, h, w, 1]
        x, y = [tf.image.extract_patches(i,
                                               [1, *self.kernel_size, 1],
                                               [1, *self.strides, 1],
                                               [1, *self.dilation_rate, 1],
                                               'VALID')
                for i in [x, y]]  # shape [1, out_h, out_w, filter_h * filter_w]
        return y, x

    @staticmethod
    def _get_pixel_values_at_point(inputs, indices):
        """get pixel values
        :param inputs:
        :param indices: shape [batch_size, H, W, I], I = filter_h * filter_w * channel_out
        :return:
        """
        y, x = indices
        # batch, h, w, n = y.get_shape().as_list()[0: 4]
        sha = tf.shape(y)

        # batch_idx = tf.reshape(tf.range(0, batch), (batch, 1, 1, 1))
        batch_idx = tf.reshape(tf.range(0, sha[0]), (sha[0], 1, 1, 1))
        
        # b = tf.tile(batch_idx, (1, h, w, n))
        # pixel_idx = tf.stack([b, y, x], axis=-1)
        b = tf.tile(batch_idx, (1, sha[1], sha[2], sha[3]))
        pixel_idx = tf.stack([b, y, x], axis=-1)
        
        return tf.gather_nd(inputs, pixel_idx)

def hw_flatten(x):
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
    
def dense_block(input_tensor, blocks):
    for i in range(blocks):
        input_tensor = conv_block(input_tensor, 32)
    return input_tensor

def conv_block(input_tensor, growth_rate):
    x1 = layers.BatchNormalization(3, epsilon = 1.001e-5)(input_tensor)
    x1 = layers.LeakyReLU()(x1)
    x1 = layers.Conv2D(4 * growth_rate, 1, use_bias = False, kernel_initializer = 'he_normal')(x1)
    x1 = layers.BatchNormalization(3, epsilon = 1.001e-5)(x1)
    x1 = layers.LeakyReLU()(x1)
    x1 = layers.Conv2D(growth_rate, 3, padding = 'same', use_bias = False, kernel_initializer = 'he_normal')(x1)
    input_tensor = layers.Concatenate(axis = 3)([input_tensor, x1])
    return input_tensor
    
def transition_block(input_tensor, reduction):
    x = layers.BatchNormalization(3, epsilon = 1.001e-5)(input_tensor)
    x = layers.LeakyReLU()(x)
    x = layers.Conv2D(int(tf.keras.backend.int_shape(x)[3] * reduction), 1, use_bias = False, kernel_initializer = 'he_normal')(x)
    x = layers.AveragePooling2D(2, strides = 2)(x)
    return x

def generator_model():
    model_in = layers.Input((720, 1280, 3))
    
    #space to depth
    s2d = tf.nn.space_to_depth(model_in, 2, data_format='NHWC')
    
    #dense block 1
    d1 = dense_block(s2d, 1)   
    d = transition_block(d1, 0.5)
    
    #dense block 2
    d2 = dense_block(d, 2)
    d = transition_block(d2, 0.5)
    
    #dense block 3
    d3 = dense_block(d, 4)

    #Self Attention Part
    sa = attention(d3)
    
    #deformable dense block 1
    ud1 = DeformableConvLayer(16, (3, 3), kernel_initializer = 'he_normal', padding = 'same', activation = tf.nn.leaky_relu)(sa)
    ud1 = layers.Conv2D(16, (1,1), kernel_initializer = 'he_normal', padding = 'same')(ud1)
    ud1 = layers.LeakyReLU()(ud1)
    ud1 = layers.concatenate([ud1, sa])

    ud2 = DeformableConvLayer(16, (3, 3), kernel_initializer = 'he_normal', padding = 'same', activation = tf.nn.leaky_relu)(ud1)
    ud2 = layers.Conv2D(16, (1, 1), kernel_initializer = 'he_normal', padding = 'same')(ud2)
    ud2 = layers.LeakyReLU()(ud2)
    ud2 = layers.concatenate([ud1, ud2])
    
    ud2 = layers.Conv2D(16, (1, 1), kernel_initializer = 'he_normal', padding = 'same')(ud2)
    ud2 = layers.LeakyReLU()(ud2)
    ud2 = layers.concatenate([ud2, sa])
    
    #upconv1
    ud2 = layers.Conv2DTranspose(8, (2, 2), strides = (2, 2), kernel_initializer = 'he_normal', padding = 'same')(ud2)
    ud2 = layers.LeakyReLU()(ud2)
    out1 = layers.concatenate([ud2, d2])
    
    #deformable dense block 2
    ud1 = DeformableConvLayer(8, (3, 3), kernel_initializer = 'he_normal', padding = 'same', activation = tf.nn.leaky_relu)(out1)
    ud1 = layers.Conv2D(8, (1,1), kernel_initializer = 'he_normal', padding = 'same')(ud1)
    ud1 = layers.LeakyReLU()(ud1)
    ud1 = layers.concatenate([ud1, out1])

    ud2 = DeformableConvLayer(8, (3, 3), kernel_initializer = 'he_normal', padding = 'same', activation = tf.nn.leaky_relu)(ud1)
    ud2 = layers.Conv2D(8, (1, 1), kernel_initializer = 'he_normal', padding = 'same')(ud2)
    ud2 = layers.LeakyReLU()(ud2)
    ud2 = layers.concatenate([ud1, ud2])
    
    ud2 = layers.Conv2D(8, (1, 1), kernel_initializer = 'he_normal', padding = 'same')(ud2)
    ud2 = layers.LeakyReLU()(ud2)
    ud2 = layers.concatenate([ud2, out1])
    
    #upconv2
    ud2 = layers.Conv2DTranspose(4, (2, 2), strides = (2, 2), kernel_initializer = 'he_normal', padding = 'same')(ud2)
    ud2 = layers.LeakyReLU()(ud2)
    out2 = layers.concatenate([ud2, d1])
    
    #deformable dense block 3
    ud1 = DeformableConvLayer(4, (3, 3), kernel_initializer = 'he_normal', padding = 'same', activation = tf.nn.leaky_relu)(out2)
    ud1 = layers.Conv2D(4, (1,1), kernel_initializer = 'he_normal', padding = 'same')(ud1)
    ud1 = layers.LeakyReLU()(ud1)
    ud1 = layers.concatenate([ud1, out2])

    ud2 = DeformableConvLayer(4, (3, 3), kernel_initializer = 'he_normal', padding = 'same', activation = tf.nn.leaky_relu)(ud1)
    ud2 = layers.Conv2D(4, (1, 1), kernel_initializer = 'he_normal', padding = 'same')(ud2)
    ud2 = layers.LeakyReLU()(ud2)
    ud2 = layers.concatenate([ud1, ud2])
    
    ud2 = layers.Conv2D(4, (1, 1), kernel_initializer = 'he_normal', padding = 'same')(ud2)
    ud2 = layers.LeakyReLU()(ud2)
    ud2 = layers.concatenate([ud2, out2])
    
    #upconv3
    ud2 = layers.Conv2DTranspose(3, (2, 2), strides = (2, 2), kernel_initializer = 'he_normal', padding = 'same')(ud2)
    ud2 = layers.LeakyReLU()(ud2)
    out3 = layers.concatenate([ud2, model_in])
    
    #depthtospace
    #d2s = tf.nn.depth_to_space(c9, 2, data_format='NHWC')
    
    #multiscale down-up
    
    #maxpool 2 x 4 x 8 x 16 layers
    max1 = layers.MaxPooling2D((2,2), padding = 'valid')(out3)
    max2 = layers.MaxPooling2D((4,4), padding = 'valid')(out3)
    max3 = layers.MaxPooling2D((8,8), padding = 'valid')(out3)
    max4 = layers.MaxPooling2D((16,16), padding = 'valid')(out3) 

    #upsample
    up1 = layers.UpSampling2D((2,2), interpolation = 'nearest')(max1)
    up2 = layers.UpSampling2D((4,4), interpolation = 'nearest')(max2)
    up3 = layers.UpSampling2D((8,8), interpolation = 'nearest')(max3)
    up4 = layers.UpSampling2D((16,16), interpolation = 'nearest')(max4)
    
    concatpool = layers.concatenate([up1, up2, up3, up4])

    u10 = layers.Conv2DTranspose(12, (1,1), kernel_initializer = 'he_normal')(concatpool)
    u10 = layers.LeakyReLU()(u10)
    model_out = layers.Conv2DTranspose(3, (1,1), activation = 'tanh')(u10)
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
    real_loss = cross_entropy( (np.random.random((real_output.shape))*0.3 + 0.7) * tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy( (np.random.random((generated_output.shape))*0.3) * tf.zeros_like(generated_output), generated_output)
    return real_loss, fake_loss

def generator_loss(fake, sharp, dis_f_loss):
    return l1_loss(fake, sharp), l2_loss(fake, sharp), Lambda1 * l1_loss(fake, sharp) + Lambda2 * l2_loss(fake, sharp) + Lambda3 * dis_f_loss

def heatmapwithoutabs(image1, image2):
    image1 = np.asarray(image1)
    image2 = np.asarray(image2)
    image1 = image1[0,:,:,:]
    image2 = image2[0,:,:,:]
    finalimage = np.zeros((image1.shape))
    finalimage = image1 - image2
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
                
                #hmapxy = heatmapwithoutabs(x_batch, y_batch)
                real_output = discriminator([x_batch, y_batch], training = True)
                fake_output = discriminator([x_batch, gen_images], training = True)
                
                dis_r_loss, dis_f_loss = discriminator_loss(real_output, fake_output)
                dis_loss = dis_r_loss + dis_f_loss
                L1, L2, gen_loss = generator_loss(gen_images, y_batch, dis_f_loss)
                
                inpimg = x_batch.numpy()
                expimg = y_batch.numpy()
                genimg = gen_images.numpy()

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
            
            gen_distape = hvd.DistributedGradientTape(gen, compression = hvd.Compression.fp16)
            dis_distape = hvd.DistributedGradientTape(dis, compression = hvd.Compression.fp16)
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
    finalimage = abs(image1 - image2)
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
Lambda1 = options.lambda1
Lambda2 = options.lambda2
Lambda3 = options.lambda3

print("Epochs are: ", EPOCHS)
print("Factors are: ", Lambda1, Lambda2, Lambda3)
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
if hvd.rank() == 0:
    test()
suffix = int(random()*10000)
np.save('./tmp/log_array_'+str(suffix)+'.npy', log_array)