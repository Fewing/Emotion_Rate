import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import os

if __name__ == '__main__':
    model = keras.models.load_model('./model/model.h5')
    test_data_list = []
    test_image_list = os.listdir("./test_image/")
    for img in test_image_list:
        url = os.path.join('./test_image/' + img)
        image = keras.preprocessing.image.load_img(url, target_size=(128, 128))
        test_data_list.append(keras.preprocessing.image.img_to_array(image))
    img_data = np.array(test_data_list)
    img_data = img_data.astype('float32')
    img_data /= 255
    test_data = img_data
    predictions = model.predict(test_data)
    i = 0
    #i为需要显示的图片序号
    plt.figure()
    plt.imshow(test_data[i])
    plt.grid(False)
    plt.xlabel(np.argmax(predictions[i]))
    plt.show()
