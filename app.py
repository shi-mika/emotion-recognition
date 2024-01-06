from flask import Flask, request, Response, render_template
from flask_cors import CORS
import cv2
import os
import numpy as np
# Libraries for Image Classification
import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression

app = Flask(__name__)
CORS(app)


TRAIN_DIR = 'train'
IMG_SIZE = 128
IMAGE_CHANNELS = 3
FIRST_NUM_CHANNEL = 32
FILTER_SIZE = 3
LR = 0.0001
PERCENT_TRAINING_DATA = 80
NUM_EPOCHS = 50
MODEL_NAME = 'facecap_cnn'

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
model.load(MODEL_NAME)

@app.route('/input-image')
def input_image():
	return render_template('input-image.html')

@app.route('/model-api', methods=['POST'])
def model_api():
	img = cv2.imdecode(np.fromstring(request.files['img'].read(), np.uint8), cv2.IMREAD_COLOR)
	img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
	# Classify image
	data = img.reshape(IMG_SIZE, IMG_SIZE, IMAGE_CHANNELS)
	data_result_float = model.predict([data])[0]
	return_str = ''
	for x in range(NUM_OUTPUT):
		# "{:.4f}".format(val)
		return_str += all_classes[x] + ' ' + "{:.4f}".format(data_result_float[x]) + '<br />'
	return return_str

if __name__ == '__main__':
	app.run(debug=True, port='8000', host='0.0.0.0', use_reloader=True)