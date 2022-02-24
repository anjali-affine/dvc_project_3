import tensorflow as tf
from tensorflow import keras
import numpy as np
import json
import sys


import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
 
import yaml

params = yaml.safe_load(open("params.yaml"))["training"]

directory = "data_12"
user_data = directory #  + "/train"
valid_data = directory #  + "/val"
test_data = directory   + "/label_book" # this can be the label book, or any other test set you create
# test_data = directory + "/test" # this can be the label book, or any other test set you create

### DO NOT MODIFY BELOW THIS LINE, THIS IS THE FIXED MODEL ###
batch_size = 8
tf.random.set_seed(123)
epochs=params["epochs"]
# epochs= 10
if __name__ == "__main__":
    train = tf.keras.preprocessing.image_dataset_from_directory(
        user_data + '/train',
        labels="inferred",
        label_mode="categorical",
        class_names=["i", "ii", "iii", "iv", "v", "vi", "vii", "viii", "ix", "x"],
        shuffle=True,
        seed=123,
        batch_size=batch_size,
        image_size=(32, 32),
    )

    valid = tf.keras.preprocessing.image_dataset_from_directory(
        user_data + '/val',
        labels="inferred",
        label_mode="categorical",
        class_names=["i", "ii", "iii", "iv", "v", "vi", "vii", "viii", "ix", "x"],
        shuffle=True,
        seed=123,
        batch_size=batch_size,
        image_size=(32, 32),
    )

    total_length = ((train.cardinality() + valid.cardinality()) * batch_size).numpy()
    if total_length > 10_000:
        print(f"Dataset size larger than 10,000. Got {total_length} examples")
        sys.exit()

    test = tf.keras.preprocessing.image_dataset_from_directory(
        test_data,
        labels="inferred",
        label_mode="categorical",
        class_names=["i", "ii", "iii", "iv", "v", "vi", "vii", "viii", "ix", "x"],
        shuffle=False,
        seed=123,
        batch_size=batch_size,
        image_size=(32, 32),
    )

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

    model.compile(
        optimizer=tf.keras.optimizers.Adam(lr=0.0001),
        loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )
    model.summary()
    loss_0, acc_0 = model.evaluate(valid, verbose=0)
    print(f"loss {loss_0}, acc {acc_0}")

    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        r"C:\Users\anjali\Desktop\dvc_project_2\project_data12\saved_model\best_model",
        monitor="val_accuracy",
        mode="max",
        save_best_only=True,
        save_weights_only=True,
    )

    history = model.fit(
        train,
        validation_data=valid,
        epochs=epochs,
        callbacks=[checkpoint],
        verbose=2
    )

    model.load_weights("saved_model/best_model")
    
    acc_=history.history['accuracy']
    test_acc_=history.history['val_accuracy']
    loss_=history.history['loss']
    test_loss_=history.history['val_loss']
    acc_ =list(acc_)
    test_acc_ =list(test_acc_)
    loss_=list(loss_)
    test_loss_=list(test_loss_)
   
    loss_train=[ {'train_loss': loss_[i] } for i in range(len(loss_)) ]
    loss_test=[ {'test_loss': test_loss_[i] } for i in range(len(test_loss_)) ]
    
    acc_train=[ {'train_acc': acc_[i]} for i in range(len(acc_)) ]
    acc_test=[ {'test_acc': test_acc_[i] } for i in range(len(test_acc_)) ]
    
    
    loss, acc = model.evaluate(valid, verbose=0)
    print(f"final loss {loss}, final acc {acc}")
    
    jsonString = json.dumps({"loss": loss, "accuracy": acc}, indent=4)
    jsonFile = open("scores.json", "w")
    jsonFile.write(jsonString)
    jsonFile.close()

    test_loss, test_acc = model.evaluate(test, verbose=0)
    print(f"test loss {test_loss}, test acc {test_acc}")

    jsonString = json.dumps({"loss_train": loss_train}, indent=4)
    jsonFile = open("loss_train.json", "w")
    jsonFile.write(jsonString)
    jsonFile.close()
    
    jsonString = json.dumps({"accuracy_train": acc_train}, indent=4)
    jsonFile = open("accuracy_train.json", "w")
    jsonFile.write(jsonString)
    jsonFile.close()

    jsonString = json.dumps({"loss_test": loss_test}, indent=4)
    jsonFile = open("loss_test.json", "w")
    jsonFile.write(jsonString)
    jsonFile.close()
    
    jsonString = json.dumps({"accuracy_test": acc_test}, indent=4)
    jsonFile = open("accuracy_test.json", "w")
    jsonFile.write(jsonString)
    jsonFile.close()