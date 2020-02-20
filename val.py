import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import os
import math

if __name__ == '__main__':
    model = keras.models.load_model('./ckpt/model.h5')
    model.summary()
    test_data_list = []
    label = []
    test_image_list = os.listdir("./val_image/")
    for img in test_image_list:
        url = os.path.join('./val_image/' + img)
        image = keras.preprocessing.image.load_img(url, target_size=(128, 128))
        test_data_list.append(keras.preprocessing.image.img_to_array(image))
        label.append(float(img.split('-')[0])-3.0)
    img_data = np.array(test_data_list)
    img_data = img_data.astype('float32')
    img_data /= 255
    label = np.array(label)
    predictions = model.predict(img_data)
    plt.scatter(label, model.predict(img_data), s=0.5)
    plt.plot(label, label)
    plt.show()
