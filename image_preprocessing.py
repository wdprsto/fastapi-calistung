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
        min_w, max_w = 15, 1200
        min_h, max_h = 15, 1200

        # find countour
        conts = self.contour_detection(processed_image.copy())

        # sort the contour
        sorted_box = self.sort_contour(conts)

        # prepare the output
        for box in sorted_box:
            (x, y, w, h) = box
            if (w >= min_w and w <= max_w) and (h >= min_h and h <= max_h):
                # the one which is processed should be the greyscaled, not inverted dilated one
                self.process_box(thresh, x, y, w, h)
        
        pixels = np.array([pixel for pixel in self.characters], dtype = 'float32')

        return pixels

    def contour_detection(self, img):
        conts = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
        conts = np.array([cv2.boundingRect(i) for i in conts])
        conts = self.filter_edge_contour(conts, img.shape)
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

        self.characters.append(normalized)

    # filter segments that touch the edge of the image
    def filter_edge_contour(self, bounding_box, image_shape):
        x, y, w, h = bounding_box[:, 0], bounding_box[:, 1], bounding_box[:, 2], bounding_box[:, 3]
        res_y, res_x = image_shape
        return bounding_box[((res_x - w - x) * (res_y - h - y) * x * y) != 0]

    def sort_contour(self, conts):
        # sort the countur, 1st top to bottom (line by line), then left to right for each line
        # sort the data from y values/top
        sort_by_line = conts[np.argsort(conts[:, 1])]

        # slice data to lines by the difference of every y where 
        # y is greater that median of the char heights
        median_h = np.median(sort_by_line[:, -1])
        diff_y = np.diff(sort_by_line[:,1])
        new_line = np.where(diff_y > median_h-5)[0] + 1 # nilai np.where perlu diubah agar pembagian linenya benar
        lines = np.array_split(sort_by_line, new_line)

        # sorted each lines from left.
        sorted_left = [line[np.argsort(line[:, 0])] for line in lines]

        sorted_box = [box for lines in sorted_left for box in lines]

        return sorted_box
