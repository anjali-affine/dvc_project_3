import tensorflow as tf
import keras 
import glob
import os
import numpy as np
import json
import sys

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


directory = "data_12"
user_data = directory #  + "/train"
valid_data = directory  + "/val"
test_data = directory   + "/label_book" # this can be the label book, or any other test set you create
# test_data = directory + "/test" # this can be the label book, or any other test set you create

### DO NOT MODIFY BELOW THIS LINE, THIS IS THE FIXED MODEL ###
batch_size = 8
tf.random.set_seed(123)


base_model = tf.keras.applications.ResNet50(
    input_shape=(32, 32, 3),
    include_top=False,
    weights=None,
)
base_model = tf.keras.Model(
    base_model.inputs, outputs=[base_model.get_layer("conv2_block3_out").output]
)

inputs = tf.keras.Input(shape=(32, 32, 3))
x = tf.keras.applications.resnet.preprocess_input(inputs)
x = base_model(x)
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dense(10)(x)
model = tf.keras.Model(inputs, x)

# model.compile(
#     optimizer=tf.keras.optimizers.Adam(lr=0.0001),
#     loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
#     metrics=["accuracy"],
# )
# 
model.load_weights("saved_model/best_model")
labels = ["i", "ii", "iii", "iv", "v", "vi", "vii", "viii", "ix", "x"]

import pandas as pd
df = pd.DataFrame()

for i in glob.glob(valid_data + "/**/*g", recursive=True):
#    print(i)
    image = tf.keras.preprocessing.image.load_img(i, target_size=(32,32))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr])  # Convert single image to a batch.
    prediction = np.argmax(model.predict(input_arr))
    head, tail = os.path.split(i)
    other, category = os.path.split(head)
    df_entry = {'file':i,'category':category,'prediction':labels[prediction]}
    df = df.append(df_entry, ignore_index=True)

for i in glob.glob(test_data + "/**/*g", recursive=True):
#    print(i)
    image = tf.keras.preprocessing.image.load_img(i, target_size=(32,32))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr])  # Convert single image to a batch.
    prediction = np.argmax(model.predict(input_arr))
    head, tail = os.path.split(i)
    other, category = os.path.split(head)
    df_entry = {'file':i,'category':category,'prediction':labels[prediction]}
#    print(df_entry)
    df = df.append(df_entry, ignore_index=True)
#    print(prediction, type(prediction))
os.makedirs('output', exist_ok=True)
df.to_csv('output\predictions.csv')
# df_incorrect = df[df['category'] != df['prediction']]
# df_incorrect.to_csv('incorrect.csv')
