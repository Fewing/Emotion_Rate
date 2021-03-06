import tensorflow as tf
from tensorflow import keras
import tensorflow_core
from sklearn.model_selection import train_test_split
import pandas as pd

import numpy as np
import matplotlib.pyplot as plt
import os
import random

if __name__ == '__main__':
    model = keras.Sequential()
    resnet = keras.applications.resnet.ResNet50(
        include_top=False, pooling='avg', input_shape=(128, 128, 3))
    model.add(resnet)
    model.add(keras.layers.Dense(1))
    model.layers[0].trainable = False
    model.summary()
    image_data_list = []
    label = []
    train_image_list = os.listdir("./train_image/")
    random.shuffle(train_image_list)
    for img in train_image_list:
        url = os.path.join('./train_image/' + img)
        image = keras.preprocessing.image.load_img(url, target_size=(128, 128))
        image_data_list.append(keras.preprocessing.image.img_to_array(image))
        label.append(float(img.split('-')[0])-3.0)
    img_data = np.array(image_data_list)
    img_data = img_data.astype('float32')
    img_data /= 255
    label = np.array(label)
    X = img_data
    Y = label
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)
    model.compile(loss='mse',
                  optimizer='adam')
    print(y_train)
    tf.keras.utils.plot_model(model, to_file='model.png')
    history = model.fit(x=X_train, y=y_train, batch_size=64,
                        epochs=30, verbose=1)
    plt.scatter(y_test, model.predict(X_test), s=0.5)
    plt.plot(y_test, y_test)
    plt.savefig("val.jpg")
    model.layers[0].trainable = True
    model.compile(loss='mse', optimizer='adam')
    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        filepath='model/model.h5', save_best_only=True, mode='min', monitor='val_loss', verbose=1)
    reduce_learning_rate = tf.keras.callbacks.ReduceLROnPlateau(monitor='loss',
                                             factor=0.1,
                                             patience=2,
                                             cooldown=2,
                                             min_lr=0.00001,
                                             verbose=1)
    history = model.fit(x=X_train,
                        y=y_train,
                        batch_size=16,
                        epochs=30,
                        validation_data=(X_test, y_test),
                        verbose=1,
                        callbacks=[checkpoint,reduce_learning_rate])
    # model.save('./model/model.h5')  # 保存模型
    best_model = keras.models.load_model('./model/model.h5')
    plt.scatter(y_test, best_model.predict(X_test), s=0.5)
    plt.plot(y_test, y_test)
    plt.savefig("val.jpg")