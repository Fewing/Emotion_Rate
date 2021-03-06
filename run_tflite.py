import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import os
import math
import time

if __name__ == '__main__':
    start_time = time.time()
    interpreter = tf.lite.Interpreter(model_path="./model/model_default.tflite")
    interpreter.allocate_tensors()

    # Get input and output tensors.
    input_details = interpreter.get_input_details()
    print(str(input_details))
    output_details = interpreter.get_output_details()
    print(str(output_details))
    image_list = os.listdir("./test_image")
    model_interpreter_time = 0
    label = []
    predictions = []
    test_data = []
    # 遍历文件
    for image in image_list:
        print('=========================')
        #label.append(float(image.split('-')[0])-3.0)
        full_path = os.path.join("./test_image", image)
        #预处理
        image = keras.preprocessing.image.load_img(full_path, target_size=(128, 128))
        image = np.array(image)
        test_data.append(image)
        image = image.astype('float32')
        image /= 255
        image_np_expanded = np.expand_dims(image, axis=0)
        image_np_expanded = image_np_expanded.astype('float32')
        # 填装数据
        model_interpreter_start_time = time.time()
        interpreter.set_tensor(input_details[0]['index'], image_np_expanded)

        # 注意注意，我要调用模型了
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])
        model_interpreter_time += time.time() - model_interpreter_start_time

        # 出来的结果去掉没用的维度
        result = np.squeeze(output_data)
        print(result)
        predictions.append(result)
    used_time = time.time() - start_time
    print('used_time:{}'.format(used_time))
    print('model_interpreter_time:{}'.format(model_interpreter_time))
    plt.figure(figsize=(10,10))
    for i in range(0,len(image_list)):
        width=int(math.sqrt(len(image_list)))+1
        plt.subplot(width,width,i+1)
        plt.xticks([])
        plt.yticks([])
        plt.imshow(test_data[i])
        plt.grid(False)
        plt.xlabel(predictions[i]+3.0)
    plt.show()

