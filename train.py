import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import os

if __name__ == '__main__':
    model = keras.Sequential([
        keras.layers.Conv2D(32, (3, 3), padding='same',
                            input_shape=(128, 128, 3)),
        keras.layers.Activation('relu'),
        keras.layers.Conv2D(32, (3, 3)),
        keras.layers.Activation('relu'),
        keras.layers.MaxPooling2D(pool_size=(2, 2)),
        keras.layers.Dropout(0.5),

        keras.layers.Flatten(),
        keras.layers.Dense(128),
        keras.layers.Activation('relu'),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(10),
        keras.layers.Activation('softmax'),
    ])
    image_data_list = []
    label = []
    train_image_list = os.listdir("./train_image/")
    for img in train_image_list:
        url = os.path.join('./train_image/' + img)
        image = keras.preprocessing.image.load_img(url, target_size=(128, 128))
        image_data_list.append(keras.preprocessing.image.img_to_array(image))
        label.append(img.split('-')[0])
    img_data = np.array(image_data_list)
    img_data = img_data.astype('float32')
    img_data /= 255
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam', metrics=['accuracy'])
    train_x = img_data
    train_y = label
    train_y = keras.utils.to_categorical(train_y)
    print(img_data.shape)
    hist = model.fit(train_x, train_y, batch_size=10, epochs=10, verbose=1)
    model.save('./model/model.h5')#保存模型