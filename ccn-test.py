import os
import numpy as np
# pip install opencv-python
import cv2
from random import shuffle
# Libraries for Image Classification
import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression

TRAIN_DIR = 'train'
IMG_SIZE = 128
IMAGE_CHANNELS = 3
FIRST_NUM_CHANNEL = 32
FILTER_SIZE = 3
LR = 0.0001
PERCENT_TRAINING_DATA = 80
NUM_EPOCHS = 50
MODEL_NAME = 'emotion_cnn'

TEST_IMG = 'sample.jpg'

def define_classes():
	all_classes = []
	for folder in os.listdir(TRAIN_DIR):
		all_classes.append(folder)
	return all_classes, len(all_classes)

def define_labels(all_classes):
	all_labels = []
	for x in range(len(all_classes)):
		all_labels.append(np.array([0. for i in range(len(all_classes))]))
		all_labels[x][x] = 1
	return all_labels

all_classes, NUM_OUTPUT = define_classes()
all_labels = define_labels(all_classes)

# Make the model
convnet = input_data(shape=[None, IMG_SIZE, IMG_SIZE, IMAGE_CHANNELS], name='input')
convnet = conv_2d(convnet, FIRST_NUM_CHANNEL, FILTER_SIZE, activation='relu')
convnet = max_pool_2d(convnet, FILTER_SIZE)
convnet = conv_2d(convnet, FIRST_NUM_CHANNEL*2, FILTER_SIZE, activation='relu')
convnet = max_pool_2d(convnet, FILTER_SIZE)
convnet = conv_2d(convnet, FIRST_NUM_CHANNEL*4, FILTER_SIZE, activation='relu')
convnet = max_pool_2d(convnet, FILTER_SIZE)
convnet = conv_2d(convnet, FIRST_NUM_CHANNEL*8, FILTER_SIZE, activation='relu')
convnet = max_pool_2d(convnet, FILTER_SIZE)
convnet = fully_connected(convnet, FIRST_NUM_CHANNEL*16, activation='relu')
convnet = dropout(convnet, 0.8)
convnet = fully_connected(convnet, NUM_OUTPUT, activation='softmax')
convnet = regression(convnet, optimizer='adam', learning_rate=LR, loss='categorical_crossentropy', name='targets')
model = tflearn.DNN(convnet, tensorboard_dir='log')

print('LOADING MODEL ' + '{}.meta'.format(MODEL_NAME))
if os.path.exists('{}.meta'.format(MODEL_NAME)):
	# Load the Model
	model.load(MODEL_NAME)
	test_img = cv2.imread(TEST_IMG)
	test_img = cv2.resize(test_img, (IMG_SIZE, IMG_SIZE))
	# Classify the image
	data = test_img.reshape(IMG_SIZE, IMG_SIZE, IMAGE_CHANNELS)
	data_res_float = model.predict([data])[0]
	for x in range(len(all_labels)):
		print(all_classes[x] + ' ' + str(data_res_float[x]*100))