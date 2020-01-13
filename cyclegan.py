#!/usr/bin/env python
# coding: utf-8

# Following code work for image to image translation with Cycle GAN.
# Baseline Code is designed by Tensorflow and it is customised by Omkar Thawakar
# as a part of research project.

from __future__ import absolute_import, division, print_function, unicode_literals

try:
  get_ipython().run_line_magic('tensorflow_version', '2.x')
except Exception:
    pass
import tensorflow as tf

import os
os.environ['CUDA_VISIBLE_DEVICES']=''
os.system('color 1')
import time
import sys
from absl import app

from matplotlib import pyplot as plt
from IPython import display

from tensorflow.keras.layers import Input, Dense, Reshape, Flatten, Dropout, Concatenate, Lambda, Add, ReLU, MaxPooling2D
import datetime
import numpy as np
from termcolor import colored, cprint

nf=8
LAMBDA = 100
IMG_WIDTH = 256
IMG_HEIGHT = 256
batch_size=1



_URL = 'https://people.eecs.berkeley.edu/~taesung_park/CycleGAN/datasets/horse2zebra.zip'

path_to_zip = tf.keras.utils.get_file('F:/Pravin/CycleGAN/horse2zebra.zip',
                                      origin=_URL,
                                      extract=True)
os.mkdir('horse2zebra')
PATH = os.path.join(os.path.dirname(path_to_zip), 'horse2zebra/')

exit(0)


def load(image_file):
    image = tf.io.read_file(image_file)
    image = tf.image.decode_jpeg(image)
    image = tf.cast(image, tf.float32)
    return image

def resize(input_image, height, width):
    input_image = tf.image.resize(input_image, [height, width],
                                method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
  
    return input_image

def random_crop(input_image):
    cropped_image = tf.image.random_crop(input_image, size=[IMG_HEIGHT, IMG_WIDTH, 3])

    return cropped_image

# normalizing the images to [-1, 1]
def normalize(input_image):
    input_image = ( input_image / 127.5) - 1

    return input_image


@tf.function()
def random_jitter(input_image):
    # resizing to 286 x 286 x 3
    input_image = resize(input_image, IMG_WIDTH, IMG_HEIGHT)

    # randomly cropping to 256 x 256 x 3
    input_image = random_crop(input_image)

    if tf.random.uniform(()) > 0.5:
        # random mirroring
        input_image = tf.image.flip_left_right(input_image)

    return input_image

def load_image_train(image_file):
	input_image = load(image_file)
	input_image = random_jitter(input_image)
	input_image = normalize(input_image)

	return input_image


def load_image_test(image_file):
    input_image = load(image_file)
    input_image = resize(input_image,IMG_HEIGHT, IMG_WIDTH)
    input_image = normalize(input_image)

    return input_image

OUTPUT_CHANNELS = 3

def batch_norm(tensor):
    return tf.keras.layers.BatchNormalization(axis=3,epsilon=1e-5, momentum=0.1, gamma_initializer=tf.random_normal_initializer(1.0, 0.02))(tensor)

##################################### Generator 1 definition ##################################################

class Generator_1:
	def __init__(self):
	    self.inputs = tf.keras.layers.Input(shape=[256,256,3])
	    self.name = 'Generator_1/'
	    print(colored('='*50,'green'))
	    print('input shape ::: ',self.inputs.shape)
	    self.generator = self.build_generator()
	    self.loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)

	def upsample_block(self, filters, size, apply_dropout=False, name='upsample'):
		initializer = tf.random_normal_initializer(0., 0.02)
		result = tf.keras.Sequential()
		result.add(
		tf.keras.layers.Conv2DTranspose(filters, size, strides=2,
		                                padding='same',
		                                kernel_initializer=initializer,
		                                use_bias=False,
		                                name = self.name+name))
		result.add(tf.keras.layers.BatchNormalization())
		if apply_dropout:
			result.add(tf.keras.layers.Dropout(0.5))
		result.add(tf.keras.layers.ReLU())
		return result

	def downsample_block(self,filters=16, size=3, apply_batchnorm=True, name='downsample'):
		initializer = tf.random_normal_initializer(0., 0.02)
		result = tf.keras.Sequential()
		result.add(
		  tf.keras.layers.Conv2D(filters, size, strides=2, padding='same',
		                         kernel_initializer=initializer, use_bias=False,
		                        name='Conv_'+name))
		result.add(tf.keras.layers.LeakyReLU())
		return result

	def Convolution(self,filters, size, apply_batchnorm=True, name='convolution'):
		initializer = tf.random_normal_initializer(0., 0.02)
		result = tf.keras.Sequential()
		result.add(
		  tf.keras.layers.Conv2D(filters, size, strides=1, padding='same',
		                         kernel_initializer=initializer, use_bias=True, name=self.name+name))
		if apply_batchnorm:
			result.add(tf.keras.layers.BatchNormalization())
		result.add(tf.keras.layers.LeakyReLU())
		return result

	def generator_loss(self, generated):
		return self.loss_object(tf.ones_like(generated), generated)

	def build_generator(self):
	    print(colored('################## Build generator 1 ##################','green'))

	    down1 = self.downsample_block(filters=16, size=3, name='DownSample1_')(self.inputs)
	    print(colored('down1 ::: {}'.format(down1.shape),'green'))

	    down2 = self.downsample_block(filters=16,size=3, name='DownSample1_')(down1)
	    print(colored('down2 ::: {}'.format(down2.shape),'green'))

	    up1 = self.upsample_block(filters=16,size=3, name='UpSample1_')(down2)
	    print(colored('up1 ::: {}'.format(up1.shape),'green'))
	    tensor = tf.keras.layers.Concatenate()([up1, down1])

	    up2 = self.upsample_block(filters=16,size=3, name='UpSample1_')(tensor)
	    print(colored('up2 ::: {}'.format(up2.shape),'green'))

	    tensor = self.Convolution(filters=3, size=3, name=self.name+'output')(up2)

	    print(colored('output ::: {}'.format(tensor.shape),'green'))
	    print(colored('='*50,'green'))
	    
	    return tf.keras.Model(inputs=self.inputs, outputs=tensor)

class Discriminator1():

	def __init__(self):
		self.input_ = tf.keras.layers.Input(shape=[256, 256, 3], name='input_image')
		self.name = 'Disriminator1/'
		self.loss_obj = tf.keras.losses.BinaryCrossentropy(from_logits=True)
		self.discriminator = self.build_discriminator()

	def conv2d(self, filters, size,stride=2,name='conv2d'):
		initializer = tf.random_normal_initializer(0., 0.02)
		result = tf.keras.Sequential()
		result.add(tf.keras.layers.Conv2D(filters, size, strides=stride,
		                                padding='same',
		                                kernel_initializer=initializer,
		                                use_bias=False,
		                                name = self.name+name))
		return result

	def downsample_block(self,filters=16, size=3, apply_batchnorm=True, name='downsample'):
		initializer = tf.random_normal_initializer(0., 0.02)
		result = tf.keras.Sequential()
		result.add(
		  tf.keras.layers.Conv2D(filters, size, strides=2, padding='same',
		                         kernel_initializer=initializer, use_bias=False,
		                        name='Conv_'+name))
		result.add(tf.keras.layers.LeakyReLU())
		return result

	def batch_norm(self, tensor):
		return tf.keras.layers.BatchNormalization(axis=3,epsilon=1e-5, momentum=0.1, gamma_initializer=tf.random_normal_initializer(1.0, 0.02))(tensor)

	def discriminator_loss(self, real, generated):
		real_loss = self.loss_obj(tf.ones_like(real), real)
		generated_loss = self.loss_obj(tf.zeros_like(generated), generated)
		total_disc_loss = real_loss + generated_loss
		return total_disc_loss * 0.5

	def build_discriminator(self):
		initializer = tf.random_normal_initializer(0., 0.02)
		print(colored('################## Build Discriminator 1 ##################','yellow'))

		down1 = self.downsample_block(filters=16, size=4, name='DownSample1_')(self.input_)
		print(colored('down1 ::: {}'.format(down1.shape),'yellow'))

		down2 = self.downsample_block(filters=16, size=4, name='DownSample2_')(down1)
		print(colored('down2 ::: {}'.format(down2.shape),'yellow'))

		down3 = self.downsample_block(filters=16, size=4, name='DownSample3_')(down2)
		print(colored('down3 ::: {}'.format(down3.shape),'yellow'))

		zero_pad1 = tf.keras.layers.ZeroPadding2D()(down3) 
		conv = tf.keras.layers.Conv2D(512, 4, strides=1,
		                            kernel_initializer=initializer,
		                            use_bias=False)(zero_pad1) 

		batchnorm1 = tf.keras.layers.BatchNormalization()(conv)

		leaky_relu = tf.keras.layers.LeakyReLU()(batchnorm1)

		zero_pad2 = tf.keras.layers.ZeroPadding2D()(leaky_relu) 

		last = tf.keras.layers.Conv2D(1, 4, strides=1,
		                            kernel_initializer=initializer)(zero_pad2) 

		return tf.keras.Model(inputs=[self.input_], outputs=last)


###############################################################################################################

##################################### Generator 2 definition ##################################################

class Generator_2:
	def __init__(self):
	    self.inputs = tf.keras.layers.Input(shape=[256,256,3])
	    self.name = 'Generator_1/'
	    print(colored('='*50,'green'))
	    print('input shape ::: ',self.inputs.shape)
	    self.generator = self.build_generator()  ### creating generator model
	    self.loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)

	def upsample_block(self, filters, size, apply_dropout=False, name='upsample'):
		initializer = tf.random_normal_initializer(0., 0.02)
		result = tf.keras.Sequential()
		result.add(
		tf.keras.layers.Conv2DTranspose(filters, size, strides=2,
		                                padding='same',
		                                kernel_initializer=initializer,
		                                use_bias=False,
		                                name = self.name+name))
		result.add(tf.keras.layers.BatchNormalization())
		if apply_dropout:
			result.add(tf.keras.layers.Dropout(0.5))
		result.add(tf.keras.layers.ReLU())
		return result

	def downsample_block(self,filters=16, size=3, apply_batchnorm=True, name='downsample'):
		initializer = tf.random_normal_initializer(0., 0.02)
		result = tf.keras.Sequential()
		result.add(
		  tf.keras.layers.Conv2D(filters, size, strides=2, padding='same',
		                         kernel_initializer=initializer, use_bias=False,
		                        name='Conv_'+name))
		result.add(tf.keras.layers.LeakyReLU())
		return result

	def Convolution(self,filters, size, apply_batchnorm=True, name='convolution'):
		initializer = tf.random_normal_initializer(0., 0.02)
		result = tf.keras.Sequential()
		result.add(
		  tf.keras.layers.Conv2D(filters, size, strides=1, padding='same',
		                         kernel_initializer=initializer, use_bias=True, name=self.name+name))
		if apply_batchnorm:
			result.add(tf.keras.layers.BatchNormalization())
		result.add(tf.keras.layers.LeakyReLU())
		return result

	def generator_loss(self, generated):
		return self.loss_object(tf.ones_like(generated), generated)

	def build_generator(self):
	    print(colored('################## Build Generator 2 ##################','cyan'))

	    down1 = self.downsample_block(filters=16, size=3, name='DownSample1_')(self.inputs)
	    print(colored('down1 ::: {}'.format(down1.shape),'cyan'))

	    down2 = self.downsample_block(filters=16,size=3, name='DownSample2_')(down1)
	    print(colored('down2 ::: {}'.format(down2.shape),'cyan'))

	    up1 = self.upsample_block(filters=16,size=3, name='UpSample1_')(down2)
	    print(colored('up1 ::: {}'.format(up1.shape),'cyan'))
	    tensor = tf.keras.layers.Concatenate()([up1, down1])

	    up2 = self.upsample_block(filters=16,size=3, name='UpSample2_')(tensor)
	    print(colored('up2 ::: {}'.format(up2.shape),'cyan'))

	    tensor = self.Convolution(filters=3, size=3, name=self.name+'output')(up2)

	    print(colored('output ::: {}'.format(tensor.shape),'cyan'))
	    print(colored('='*50,'cyan'))
	    
	    return tf.keras.Model(inputs=self.inputs, outputs=tensor)

class Discriminator2():

	def __init__(self):
		self.input_ = tf.keras.layers.Input(shape=[256, 256, 3], name='input_image')
		self.name = 'Disriminator2/'
		self.loss_obj = tf.keras.losses.BinaryCrossentropy(from_logits=True)
		self.discriminator = self.build_discriminator()

	def conv2d(self, filters, size,stride=2,name='conv2d'):
		initializer = tf.random_normal_initializer(0., 0.02)
		result = tf.keras.Sequential()
		result.add(tf.keras.layers.Conv2D(filters, size, strides=stride,
		                                padding='same',
		                                kernel_initializer=initializer,
		                                use_bias=False,
		                                name = self.name+name))
		return result

	def downsample_block(self,filters=16, size=3, apply_batchnorm=True, name='downsample'):
		initializer = tf.random_normal_initializer(0., 0.02)
		result = tf.keras.Sequential()
		result.add(
		  tf.keras.layers.Conv2D(filters, size, strides=2, padding='same',
		                         kernel_initializer=initializer, use_bias=False,
		                        name='Conv_'+name))
		result.add(tf.keras.layers.LeakyReLU())
		return result

	def batch_norm(self, tensor):
		return tf.keras.layers.BatchNormalization(axis=3,epsilon=1e-5, momentum=0.1, gamma_initializer=tf.random_normal_initializer(1.0, 0.02))(tensor)

	def discriminator_loss(self, real, generated):
		real_loss = self.loss_obj(tf.ones_like(real), real)

		generated_loss = self.loss_obj(tf.zeros_like(generated), generated)

		total_disc_loss = real_loss + generated_loss

		return total_disc_loss * 0.5

	def build_discriminator(self):
		initializer = tf.random_normal_initializer(0., 0.02)
		print(colored('################## Build Discriminator 1 ##################','magenta'))

		down1 = self.downsample_block(filters=16, size=4, name='DownSample1_')(self.input_)
		print(colored('down1 ::: {}'.format(down1.shape),'magenta'))

		down2 = self.downsample_block(filters=16, size=4, name='DownSample2_')(down1)
		print(colored('down2 ::: {}'.format(down2.shape),'magenta'))

		down3 = self.downsample_block(filters=16, size=4, name='DownSample3_')(down2)
		print(colored('down3 ::: {}'.format(down3.shape),'magenta'))

		zero_pad1 = tf.keras.layers.ZeroPadding2D()(down3) 
		conv = tf.keras.layers.Conv2D(512, 4, strides=1,
		                            kernel_initializer=initializer,
		                            use_bias=False)(zero_pad1) 

		batchnorm1 = tf.keras.layers.BatchNormalization()(conv)

		leaky_relu = tf.keras.layers.LeakyReLU()(batchnorm1)

		zero_pad2 = tf.keras.layers.ZeroPadding2D()(leaky_relu) 

		last = tf.keras.layers.Conv2D(1, 4, strides=1,
		                            kernel_initializer=initializer)(zero_pad2) 

		return tf.keras.Model(inputs=[self.input_], outputs=last) 

class GAN(object):
	"""GAN class.
	Args:
	epochs: Number of epochs.
	path: path to folder containing images (training and testing)..
	mode: (train, test).
	output_path : output path for saving model
	"""
	def __init__(self, epochs,path,mode,output_path):
		self.epochs = epochs
		self.path = path
		self.output_path = output_path
		os.path.join(self.output_path)
		self.lambda_value = 100
		self.gen1 = Generator_1()
		self.generator1 = self.gen1.generator
		self.print_info(self.generator1, 'Generator 1')

		self.disc1 = Discriminator1()
		self.discriminator1 = self.disc1.discriminator
		self.print_info(self.discriminator1, 'discriminator 1')

		self.disc2 = Discriminator2()
		self.discriminator2 = self.disc2.discriminator
		self.print_info(self.discriminator2, 'discriminator 2')

		self.gen2 = Generator_2()
		self.generator2 = self.gen2.generator
		self.print_info(self.generator2, 'Generator 2')

		self.generator1_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
		self.discriminator1_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

		self.generator2_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
		self.discriminator2_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

		self.checkpoint_dir1 = self.output_path + './training_checkpoints/' + 'gen1'
		self.checkpoint_prefix1 = os.path.join(self.checkpoint_dir1, "ckpt")
		self.checkpoint1 = tf.train.Checkpoint(generator1_optimizer=self.generator1_optimizer,
		                                 discriminator1_optimizer=self.discriminator1_optimizer,
		                                 generator1=self.generator1,
		                                 discriminator1=self.discriminator1,
		                                 )

		self.checkpoint_dir2 = self.output_path + './training_checkpoints/' + 'gen2'
		self.checkpoint_prefix2 = os.path.join(self.checkpoint_dir2, "ckpt")
		self.checkpoint2 = tf.train.Checkpoint(generator2_optimizer=self.generator2_optimizer,
		                                 discriminator2_optimizer=self.discriminator2_optimizer,
		                                 generator2=self.generator2,
		                                 discriminator2=self.discriminator2)

		log_dir = self.output_path + "/logs/"
		self.summary_writer = tf.summary.create_file_writer(log_dir + "fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

	def calc_cycle_loss(self, real_image, cycled_image):
		loss1 = tf.reduce_mean(tf.abs(real_image - cycled_image))
		return LAMBDA * loss1

	def identity_loss(self, real_image, same_image):
		loss = tf.reduce_mean(tf.abs(real_image - same_image))
		return LAMBDA * 0.5 * loss

	def generate_images(self, test_input, number, mode='train'):
		if mode == 'train':
		    gen1_prediction = self.generator1(test_input, training=True)
		    display_list = [test_input[0], gen1_prediction[0]]
		    image = np.hstack([img for img in display_list])
		    try :
		        os.mkdir(self.output_path+'/{}/'.format(mode))
		    except:
		        pass
		    plt.imsave(self.output_path+'/{}/{}_.png'.format(mode,number), np.array((image * 0.5 + 0.5)*255, dtype='uint8'))
		elif mode == 'test' :
		    gen1_prediction = self.generator1(test_input, training=True)
		    display_list = [test_input[0], gen1_prediction[0]]
		    image = np.hstack([img for img in display_list])
		    try :
		        os.mkdir(self.output_path+'/{}'.format(mode))
		    except:
		        pass
		    plt.imsave(self.output_path+'/{}/{}_.png'.format(mode,number), np.array((image * 0.5 + 0.5)*255, dtype='uint8'))
		else:
		    print('Enter valid mode eighter [!]train or [!]test')
		    exit(0)

	def print_info(self,object, name):
		print('='*50)
		text = colored('Total Trainable parameters of {} are :: {}'.format(object.count_params(), name), 'red', attrs=['reverse','blink'])
		print(text)
		print('='*50)

	def train_step(self,real_x, real_y, epoch):
		with tf.GradientTape() as gen_tape1, tf.GradientTape() as disc_tape1, tf.GradientTape() as gen_tape2, tf.GradientTape() as disc_tape2 :
			# Generator G translates X -> Y.
			# Generator F translates Y -> X.
			fake_y = self.generator1(real_x, training=True)
			cycled_x = self.generator2(fake_y, training=True)

			fake_x = self.generator2(real_y, training=True)
			cycled_y = self.generator1(fake_x, training=True)

			# same_x and same_y are used for identity loss.
			same_x = self.generator2(real_x, training=True)
			same_y = self.generator1(real_y, training=True)

			disc_real_x = self.discriminator1(real_x, training=True)
			disc_real_y = self.discriminator2(real_y, training=True)

			disc_fake_x = self.discriminator1(fake_x, training=True)
			disc_fake_y = self.discriminator2(fake_y, training=True)

			# calculate the loss
			gen_g_loss = self.gen1.generator_loss(disc_fake_y)
			gen_f_loss = self.gen2.generator_loss(disc_fake_x)

			total_cycle_loss = self.calc_cycle_loss(real_x, cycled_x) + self.calc_cycle_loss(real_y, cycled_y)

			# Total generator loss = adversarial loss + cycle loss
			total_gen_g_loss = gen_g_loss + total_cycle_loss + self.identity_loss(real_y, same_y)
			total_gen_f_loss = gen_f_loss + total_cycle_loss + self.identity_loss(real_x, same_x)

			disc_x_loss = self.disc1.discriminator_loss(disc_real_x, disc_fake_x)
			disc_y_loss = self.disc2.discriminator_loss(disc_real_y, disc_fake_y)
		  
		# Calculate the gradients for generator and discriminator
		generator_g_gradients = gen_tape1.gradient(total_gen_g_loss, 
		                                    self.generator1.trainable_variables)
		generator_f_gradients = gen_tape2.gradient(total_gen_f_loss, 
		                                    self.generator2.trainable_variables)

		discriminator_x_gradients = disc_tape1.gradient(disc_x_loss, 
		                                        self.discriminator1.trainable_variables)
		discriminator_y_gradients = disc_tape2.gradient(disc_y_loss, 
		                                        self.discriminator2.trainable_variables)

		# Apply the gradients to the optimizer
		self.generator1_optimizer.apply_gradients(zip(generator_g_gradients, 
		                                        self.generator1.trainable_variables))

		self.generator2_optimizer.apply_gradients(zip(generator_f_gradients, 
		                                        self.generator2.trainable_variables))

		self.discriminator1_optimizer.apply_gradients(zip(discriminator_x_gradients,
		                                            self.discriminator1.trainable_variables))

		self.discriminator2_optimizer.apply_gradients(zip(discriminator_y_gradients,
		                                            self.discriminator2.trainable_variables))

		with self.summary_writer.as_default():
		    tf.summary.scalar('gen_g_loss', gen_g_loss, step=epoch)
		    tf.summary.scalar('total_gen_g_loss', total_gen_g_loss, step=epoch)
		    tf.summary.scalar('disc_x_loss', disc_x_loss, step=epoch)

		    tf.summary.scalar('gen_f_loss', gen_f_loss, step=epoch)
		    tf.summary.scalar('total_gen_f_loss', total_gen_f_loss, step=epoch)
		    tf.summary.scalar('disc_y_loss', disc_y_loss, step=epoch)

		    tf.summary.scalar('total_cycle_loss', total_cycle_loss, step=epoch)

		outputs = {
		            'gen_g_loss' : gen_g_loss , 
		            'total_gen_g_loss' : total_gen_g_loss, 
		            'disc_x_loss' : disc_x_loss,
		            'gen_f_loss' : gen_f_loss,
		            'total_gen_f_loss' : total_gen_f_loss , 
		            'disc_y_loss' : disc_y_loss, 
		            'total_cycle_loss' : total_cycle_loss, 
		        }

		return outputs

	def fit(self, train_horses, train_zebras, epochs):
		for epoch in range(epochs):
			start = time.time()
			for example_input in train_horses.take(1):
				self.generate_images(example_input,epoch)
			print(colored("Epoch: {}".format(epoch),'green',attrs=['reverse','blink']))
			n = 0
			for (i, image_x), (j, image_y) in zip(train_horses.enumerate(), train_zebras.enumerate()):
				outputs = self.train_step(image_x, image_y, epoch)
				if n % 10 == 0:
				  print ('.', end='')
				n+=1
			print()
			print('='*50)
			print(colored('[!] gen_g_loss :: {}'.format(outputs['gen_g_loss']),'green'))
			print(colored('[!] total_gen_g_loss :: {}'.format(outputs['total_gen_g_loss']),'green'))
			print(colored('[!] disc_x_loss :: {}'.format(outputs['disc_x_loss']),'green'))

			print(colored('[!] gen_f_loss :: {}'.format(outputs['gen_f_loss']),'yellow'))
			print(colored('[!] total_gen_f_loss :: {}'.format(outputs['total_gen_f_loss']),'yellow'))
			print(colored('[!] disc_y_loss :: {}'.format(outputs['disc_y_loss']),'yellow'))

			print(colored('[!] total_cycle_loss :: {}'.format(outputs['total_cycle_loss']),'magenta'))
			print('='*50)
			# Using a consistent image (sample_horse) so that the progress of the model
			# is clearly visible.
		
			# saving (checkpoint) the model every 20 epochs
			if (epoch + 1) % 5 == 0:
				self.checkpoint1.save(file_prefix = self.checkpoint_prefix1)
				self.checkpoint2.save(file_prefix = self.checkpoint_prefix2)
				print ('Saving checkpoint for epoch {} '.format(epoch+1,))
			print ('Time taken for epoch {} is {} sec\n'.format(epoch + 1,
			                                                  time.time()-start))

		self.checkpoint1.save(file_prefix = self.checkpoint_prefix1)
		self.checkpoint2.save(file_prefix = self.checkpoint_prefix2)

	def test(self, dataset):
		self.checkpoint1.restore(tf.train.latest_checkpoint(self.checkpoint_prefix1)) 
		self.checkpoint2.restore(tf.train.latest_checkpoint(self.checkpoint_prefix2)) 
		text = colored('Checkpoint restored !!!','magenta')
		print(text)
		print(colored('='*50,'magenta'))
		for n, (example_input) in dataset.enumerate():
		    self.generate_images(example_input, n, mode='test')
		print(colored("Model Tested Successfully !!!!! ",'green',attrs=['reverse','blink'])) 


def run_main(argv):
	del argv
	kwargs = {'epochs': 500, 
			'path': 'dataset',
				'mode':'train', 
				'output_path':'Exp_1',
				'batch_size':1,
	        }
	main(**kwargs)

def get_train_dataset(path, folder, img_type='*.jpg'):
	_dataset = tf.data.Dataset.list_files('{}/{}/{}'.format(path,folder,img_type))
	_dataset = _dataset.map(load_image_train)
	_dataset = _dataset.shuffle(1)
	_dataset = _dataset.batch(batch_size)

	return _dataset

def get_test_dataset(path, folder, img_type='*.jpg'):
	_dataset = tf.data.Dataset.list_files('{}/{}/{}'.format(path,folder,img_type))
	_dataset = _dataset.map(load_image_test)
	_dataset = _dataset.shuffle(1)
	_dataset = _dataset.batch(batch_size)

	return _dataset

def main(epochs, path,mode,output_path,batch_size):

	gan = GAN(epochs,path,mode,output_path)
	if mode=='train':
		################# train Horses ###################
		train_horses = get_train_dataset(path, 'trainA','*.jpg')
		################# train Zebras ###################
		train_zebras = get_train_dataset(path, 'trainB','*.jpg')

		################ train Horses ##################
		test_horses = get_test_dataset(path, 'testA','*.jpg')
		################ train Zebras ##################
		test_zebras = get_test_dataset(path, 'testB','*.jpg')
		
		print('Training !!!!!')

		gan.fit(train_horses, train_zebras,epochs)

	elif mode=='test':
		################ train Horses ##################
		test_horses = get_test_dataset(path, 'testA','*.jpg')
		
		gan.test(test_horses) 

if __name__ == '__main__':
  app.run(run_main)