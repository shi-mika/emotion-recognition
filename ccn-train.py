##2to read the directory
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
##search the train folders of the images
TRAIN_DIR = 'train'

IMG_SIZE = 128##pixels
IMAGE_CHANNELS = 3 ##RGB
FIRST_NUM_CHANNEL = 32##channels that we want(layers)
FILTER_SIZE = 3 ##
LR = 0.0001
PERCENT_TRAINING_DATA = 80 #o index to 79 index
NUM_EPOCHS = 50
MODEL_NAME = 'emotion_cnn'
##1. read all the classes in the train folder
def define_classes():
	all_classes = []
	for folder in os.listdir(TRAIN_DIR):
		all_classes.append(folder)
	return all_classes, len(all_classes) ## you can return two arguments, length how many classes

##4.get the images inside of the folder
def define_labels(all_classes):
	all_labels = []
	##length of the classes that will be passing
	for x in range(len(all_classes)):
		##set a value 0 into the array
		all_labels.append(np.array([0. for i in range(len(all_classes))]))
		all_labels[x][x] = 1
	return all_labels

##5. get all the images in the one hat encoded [[image],[output]]
def create_train_data(all_classes, all_labels):
	training_data = []
	for label_index, specific_class in enumerate(all_classes):
		##the name of the directory + specific folder
		current_dir = TRAIN_DIR + '/' + specific_class
		##output correct directory
		print('Reading directory of ' + current_dir)

		##output the correct file in the folder
		for img_filename in os.listdir(current_dir):
			path = os.path.join(current_dir, img_filename)
			##actual image will going to read from the path
			img = cv2.imread(path) ##reading as colored image
			img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))##resize the image width and the height
			##append
			training_data.append([np.array(img), np.array(all_labels[label_index])])
	shuffle(training_data)##read randomly
	return training_data ##reading from the top to bottom,best to shuffle whatever we got
##3.all classes
all_classes, NUM_OUTPUT = define_classes()
##you can print
all_labels = define_labels(all_classes)
##7.you may print the training data
training_data = create_train_data(all_classes, all_labels)

# 8. Make the model
##first is to input
convnet = input_data(shape=[None, IMG_SIZE, IMG_SIZE, IMAGE_CHANNELS], name='input')
##define the num channel
convnet = conv_2d(convnet, FIRST_NUM_CHANNEL, FILTER_SIZE, activation='relu')
##filter size
convnet = max_pool_2d(convnet, FILTER_SIZE)
##another convolution
convnet = conv_2d(convnet, FIRST_NUM_CHANNEL*2, FILTER_SIZE, activation='relu')
convnet = max_pool_2d(convnet, FILTER_SIZE)
##another
convnet = conv_2d(convnet, FIRST_NUM_CHANNEL*4, FILTER_SIZE, activation='relu')
convnet = max_pool_2d(convnet, FILTER_SIZE)
##another
convnet = conv_2d(convnet, FIRST_NUM_CHANNEL*8, FILTER_SIZE, activation='relu')
convnet = max_pool_2d(convnet, FILTER_SIZE)
#connect all the layer
convnet = fully_connected(convnet, FIRST_NUM_CHANNEL*16, activation='relu')
##training 80%. already memorized the pixels, we don't the filters to be trainable,we will be more smarter
convnet = dropout(convnet, 0.8) ##weights are trainable

#another fully connected layer
convnet = fully_connected(convnet, NUM_OUTPUT, activation='softmax')
convnet = regression(convnet, optimizer='adam', learning_rate=LR, loss='categorical_crossentropy', name='targets')
model = tflearn.DNN(convnet, tensorboard_dir='log')

# Define training and testing data

##example 80th image to the last; casting the length of the training data
train = training_data[:int(len(training_data)*(PERCENT_TRAINING_DATA/100))]
test = training_data[-int(len(training_data)*(PERCENT_TRAINING_DATA/100)):]
X = np.array([i[0] for i in train]).reshape(-1, IMG_SIZE, IMG_SIZE, IMAGE_CHANNELS)
Y = [i[1] for i in train]
test_x = np.array([i[0] for i in test]).reshape(-1, IMG_SIZE, IMG_SIZE, IMAGE_CHANNELS)
test_y = [i[1] for i in test]

# Train the model
model.fit(
	{'input': X}, ##image
	{'targets': Y}, #output
	n_epoch=NUM_EPOCHS,
	validation_set=({'input': test_x}, {'targets': test_y}),
	show_metric=True
)

# Save the immediately
model.save(MODEL_NAME)