import tensorflow as tf
from tensorflow import keras
#将keras模型转换为tflite模型
if __name__ == '__main__':
    model = keras.models.load_model('./model/model.h5')
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_SIZE]
    tflite_model = converter.convert()
    open("./model/model.tflite", "wb").write(tflite_model)