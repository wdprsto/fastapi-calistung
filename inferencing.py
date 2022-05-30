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

        (pixels, boxes) = self.processed_data
    
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
            for (prediction, (x, y, w, h)) in zip(output_data, boxes):
                i = np.argmax(prediction)
                character = classLabels[i]
                predicted_data.append([character, x, y])
        
            # first, group by the y value since we want to group per line
            a = np.array([i[2] for i in predicted_data])# or np.array(d)[:,n] if all the elements of d have the same shape
            b,c = np.where(np.abs(a-a[:,None]) < 20)# I used a maximum distance of 20 between character in group

            e = set(tuple(k[1] for k in j) for i,j in groupby(zip(b,c), key=lambda x:x[0]))

            grouped_list = [[predicted_data[j] for j in i] for i in e]

            grouped_list.sort(key=lambda c: c[0][2])

            output = []
            _ = [[output.append(x[0]) for x in y] for y in grouped_list]
            output_text = "".join(output)
        # PROCESS ONLY 1 CHARACTER
        else:
            i = np.argmax(output_data[0])
            output_text = classLabels[i]
            
        return output_text