import tensorflow as tf
import numpy as np
from itertools import groupby
from image_preprocessing import TextRecognizer

class TFLiteInferencer:
    def __init__(self, image):
        recognizer = TextRecognizer(image)
        self.processed_data = recognizer.recognize_text()
    
    def predict(self):
        # clear backend session of tf
        tf.keras.backend.clear_session()

        pixels = self.processed_data
    
        # load tflite model
        interpreter = tf.lite.Interpreter(model_path="model.tflite")
        
        # Get input and output tensors.
        input_details = interpreter.get_input_details()[0]['index']
        output_details = interpreter.get_output_details()[0]['index']

        interpreter.allocate_tensors()

        output_data = []
        for i in range(len(pixels)) :
            interpreter.set_tensor(input_details, [pixels[i]])

            # run the inference
            interpreter.invoke()

            # output_details[0]['index'] = the index which provides the input
            output_data.append(interpreter.get_tensor(output_details))

        classLabels = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        # PROCESS TEXT, SINGLE OR MULTIPLE LINE
        if len(output_data)>1:
            predicted_data = []
            for prediction in output_data:
                i = np.argmax(prediction)
                character = classLabels[i]
                predicted_data.append(character)

            output_text = "".join(predicted_data)
            
        # PROCESS ONLY 1 CHARACTER
        else:
            i = np.argmax(output_data[0])
            output_text = classLabels[i]
            
        return output_text