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


TRAIN_DIR = 'train'
IMG_SIZE = 128
IMAGE_CHANNELS = 3
FIRST_NUM_CHANNEL = 32
FILTER_SIZE = 3
LR = 0.0001
PERCENT_TRAINING_DATA = 80
NUM_EPOCHS = 50
MODEL_NAME = 'emotion_cnn'

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

import tensorflow as tf

# Load the TensorFlow model
model = tf.keras.models.load_model('D:\\Downloads\\regression\\emotion-final\\emotion_cnn')
# Convert the model to TensorFlow Lite format
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Save the TensorFlow Lite model
with open('converted_model.tflite', 'wb') as f:
    f.write(tflite_model)

from flask import Flask, request, jsonify


app = Flask(__name__)

@app.route('/classify_image', methods=['POST'])
def classify_image():
    uploaded_file = request.files['file']
    file_path = 'path/to/save/' + uploaded_file.filename
    uploaded_file.save(file_path)

    test_img = cv2.imread(file_path)
    test_img = cv2.resize(test_img, (IMG_SIZE, IMG_SIZE))
    data = test_img.reshape(IMG_SIZE, IMG_SIZE, IMAGE_CHANNELS)
    data_res_float = model.predict([data])[0]

    result = {}
    for x in range(len(all_labels)):
        result[all_classes[x]] = float(data_res_float[x] * 100)

    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)