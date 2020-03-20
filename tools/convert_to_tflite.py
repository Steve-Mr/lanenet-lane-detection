import cv2
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np


def convert_from_frozen_graph():
    output_node_names = [
        'lanenet/final_binary_output',
        'lanenet/final_pixel_embedding_output'
    ]
    convert =tf.compat.v1.lite.TFLiteConverter.from_frozen_graph("./mnn_project/lanenet.pb", ["lanenet/input_tensor"],
                                                        output_node_names,
                                                        input_shapes={"lanenet/input_tensor": [1, 256, 512, 3]})
    convert.post_training_quantize = True
    tflite_model = convert.convert()
    open("./mnn_project/quantized_model.tflite", "wb").write(tflite_model)
    print("finish")


def tflite_model_test(model_path, image_path):
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    image = cv2.resize(image, (512, 256), interpolation=cv2.INTER_LINEAR)
    image = image / 127.5 - 1.0  # 归一化 (只归一未改变维数)
    image = image.reshape(1, 256, 512, 3)
    image = image.astype((np.float32))

    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    print(input_details)
    print(output_details)

    interpreter.set_tensor(input_details[0]['index'], image)

    interpreter.invoke()

    final_binary_output = interpreter.get_tensor(output_details[0]['index'])
    final_embedding_output = interpreter.get_tensor(output_details[1]['index'])

    print("binary")
    print(final_binary_output)
    print("instance")
    # final_embedding_output = (final_embedding_output + 1.0) * 127.5
    print(final_embedding_output)

    plt.figure('final_binary_output')
    plt.imshow(final_binary_output[0])
    plt.figure('final_instance_output')
    plt.imshow(final_embedding_output[0])

    plt.show()


if __name__ == '__main__':

    # convert_from_frozen_graph()
    image_path = "./data/tusimple_test_image/2.jpg"
    model_path = "./mnn_project/quantized_model.tflite"
    tflite_model_test(model_path, image_path)
