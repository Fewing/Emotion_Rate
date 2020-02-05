import tensorflow as tf
from tensorflow import keras
import tensorflow_core
from tensorflow.keras import layers

import numpy as np
import matplotlib.pyplot as plt
import os
import random

if __name__ == '__main__':
    model = keras.Sequential()
    '''
    resnet = keras.applications.ResNet50(include_top=False, pooling='avg', input_shape=(128,128,3))    
    model.add(resnet)
    model.add(layers.Dense(1))
    '''
    model.add(layers.Conv2D(24, (3, 3), padding='same', input_shape=(128, 128, 3),use_bias=True,kernel_initializer='random_normal',bias_initializer='random_normal'))
    model.add(layers.Activation('relu'))
    model.add(layers.Conv2D(96, (3, 3),use_bias=True,kernel_initializer='random_normal',bias_initializer='random_normal'))
    model.add(layers.Activation('relu'))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(layers.Dropout(0.5))

    model.add(layers.Flatten())
    model.add(layers.Dense(192,kernel_initializer='random_normal',bias_initializer='random_normal'))
    model.add(layers.Activation('relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(1))
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
    train_x = img_data
    train_y = label
    print(train_y)
    opt=keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    model.compile(loss='mse',
                  optimizer=opt)
    hist = model.fit(x=train_x, y=train_y, batch_size=10,
                     epochs=20, verbose=1, validation_split=0.2)
    model.save('./model/model.h5')  # 保存模型
