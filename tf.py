#### This python file is to check whether tf lite model is working fine or not
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt 
import cv2
size=(150,150)
img = cv2.imread("cat.57.JPG",cv2.IMREAD_COLOR)
img=cv2.resize(img,(150,150))
img = np.array(img, dtype="float32")
img=img/255.0
plt.imshow(img)
plt.show()
img = np.reshape(img, (1,150,150,3))
categories=['cat','dog']

# Load the TFLite model and allocate tensors.
interpreter = tf.lite.Interpreter(model_path="model.tflite")
interpreter.allocate_tensors()

# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Test the model on random input data.
input_shape = input_details[0]['shape']

print("*"*50, input_details)
interpreter.set_tensor(input_details[0]['index'], img)

interpreter.invoke()

# The function `get_tensor()` returns a copy of the tensor data.
# Use `tensor()` in order to get a pointer to the tensor.
output_data = interpreter.get_tensor(output_details[0]['index'])
print(categories[(np.argmax(output_data))])


