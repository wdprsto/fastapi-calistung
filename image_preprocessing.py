import cv2
import numpy as np
import imutils
from imutils.contours import sort_contours

class RequestImageConverter:
    def __init__(self, file):
        self.file = file

    def convert(self) :
        numpy_image = np.fromstring(self.file, dtype='uint8')
        image = cv2.imdecode(numpy_image, cv2.IMREAD_COLOR)
        return image

# class ImagePreprocessor:
#     def __init__(self, image):
#         self.image = image

#     def process(self):
#         # parameters
#         identitiy_matrix_shape = (3, 3) #used in dilatation

#         # make the image color grey
#         gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)

#         # invert the image color
#         _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)

#         # dilatate the image
#         dilated = cv2.dilate(thresh, np.ones(identitiy_matrix_shape, np.uint8))

#         edges = cv2.Canny(dilated, 40, 150)

#         # dilatate the image
#         processed_image = cv2.dilate(edges, np.ones(identitiy_matrix_shape))

#         return processed_image

class TextRecognizer:
    def __init__(self, image):
        self.image = image
        self.characters = []

    def recognize_text(self):
        '''
        STEP 1: IMAGE PREPROCESSING
        '''
        # parameters
        identitiy_matrix_shape = (3, 3) #used in dilatation

        # make the image color grey
        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
            
        # invert the image color
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)

        # dilatate the image
        dilated = cv2.dilate(thresh, np.ones(identitiy_matrix_shape))

        edges = cv2.Canny(dilated, 40, 150)

        # dilatate the image
        processed_image = cv2.dilate(edges, np.ones(identitiy_matrix_shape))

        '''
        STEP 2: TEXT RECOGNIZER
        '''
        # parameters
        min_w, max_w = 4, 400
        min_h, max_h = 14, 400

        # find countour
        conts = self.contour_detection(processed_image.copy())

        # prepare the output
        for c in conts:
            (x, y, w, h) = cv2.boundingRect(c)
            if (w >= min_w and w <= max_w) and (h >= min_h and h <= max_h):
                # the one which is processed should be the greyscaled, not inverted dilated one
                self.process_box(thresh, x, y, w, h)
        
        boxes = np.array([box[1] for box in self.characters])
        pixels = np.array([pixel[0] for pixel in self.characters], dtype = 'float32')

        return (pixels, boxes)

    def contour_detection(self, img):
        conts = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        conts = imutils.grab_contours(conts)
        conts = sort_contours(conts, method = 'left-to-right')[0]
        return conts

    # Extract Range of Interest (ROI)
    def extract_roi(self, conts, x, y, w, h):
        roi = conts[y:y + h, x:x + w]
        return roi

    #Resize the Image
    def resize_img(self, img, w, h):
        if w > h:
            resized = imutils.resize(img, width = 28)
        else:
            resized = imutils.resize(img, height = 28)

        (h, w) = resized.shape
        dX = int(max(0, 28 - w) / 2.0)
        dY = int(max(0, 28 - h) / 2.0)

        filled = cv2.copyMakeBorder(resized, top=dY, bottom=dY, right=dX, left=dX, borderType=cv2.BORDER_CONSTANT, value = (0,0,0))
        filled = cv2.resize(filled, (28,28))
        return filled

    def normalization(self, img):
        # normalize the image, and expand the dimension so it click our model dims
        img = img.astype('float32') / 255.0
        img = np.expand_dims(img, axis = -1)
        return img

    def process_box(self, img, x, y, w, h):
        roi = self.extract_roi(img, x, y, w, h)
        (h, w) = roi.shape
        resized = self.resize_img(roi, w, h)
        normalized = self.normalization(resized)

        self.characters.append((normalized, (x, y, w, h)))
