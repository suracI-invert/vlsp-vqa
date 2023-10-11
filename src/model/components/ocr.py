import matplotlib.pyplot as plt
from PIL import Image, ImageOps, ImageFilter
import numpy as np
import cv2
import easyocr

from vietocr.tool.predictor import Predictor
from vietocr.tool.config import Cfg

class OCR(object):
    def __init__(self):
        # init easyocr
        self.reader = easyocr.Reader(['vi'], gpu=True) 

        self.ocr_config = Cfg.load_config_from_name('vgg_transformer')
        self.ocr_config['cnn']['pretrained']= True
        self.ocr_config['device'] = 'cuda:0'

        # init text detector
        self.ocr_detector = Predictor(self.ocr_config)
  
    
    def __call__(self, image_path):
        img = cv2.imread(image_path)
        # if img is None:
        #     print(image_path)
        #plt.imshow(img)
        
        # boost contrast
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        balanced_image = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        balanced_image[:, :, 0] = clahe.apply(balanced_image[:, :, 0])
        balanced_image = cv2.cvtColor(balanced_image, cv2.COLOR_LAB2BGR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        balanced_image = cv2.equalizeHist(img)
        

        img = balanced_image
  
        text_boxes = self.reader.readtext(img)
        
        def get_center(bbox):
            bbox = np.array(bbox)
            x1, y1, x2, y2 = bbox
            return ([(x1[0] + x2[0]) // 2, (x1[1] + x2[1]) // 2], 
                    [(y1[0] + y2[0]) // 2, (y1[1] + y2[1]) // 2])

        # Sort bounding boxes based on the proximity of their center points
        sorted_text_boxes = sorted(text_boxes, key=lambda box: get_center(box[0]))

        # cropped_images = []

        threshold = 0.01

        def convert_uint(i):
            if i < 0:
                return 0
            return int(i)
        # print(len(sorted_text_boxes))
        token_list = []
        for text_box in sorted_text_boxes:
            bbox, _, score = text_box
            if len(bbox) >= 4 and score > threshold:
                pt1 = (convert_uint(bbox[0][0]), convert_uint(bbox[0][1]))
                pt2 = (convert_uint(bbox[2][0]), convert_uint(bbox[2][1]))
                # print(img[pt1[1]:pt2[1], pt1[0]:pt2[0]])
                cropped_image = img[pt1[1]:pt2[1], pt1[0]:pt2[0]]
                # print(len(cropped_image))
                # print(len(cropped_image[12]))
                if len(cropped_image) == 0 or cropped_image.shape[0] == 0 or cropped_image.shape[1] == 0:
                    continue

                ## pytereesact
                #text = pytesseract_ocr(cropped_image)
                #print(text)
                ## vietocr
                im = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB)
                image_pil = Image.fromarray(im)
                image_pil = ImageOps.grayscale(image_pil)
                s = self.ocr_detector.predict(image_pil)
                token_list.append(s)

        return token_list



# def text_tokens(image_path):
#     img = cv2.imread(image_path)
#     #plt.imshow(img)
    
#     # init text detector
#     if 'ocr_detector' not in globals():
#         ocr_config = Cfg.load_config_from_name('vgg_transformer')
#         # config['weights'] = './weights/transformerocr.pth'
#         ocr_config['cnn']['pretrained']= True
#         ocr_config['device'] = 'cuda:0'
#         ocr_detector = Predictor(ocr_config)
    
#     # boost contrast
#     clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
#     balanced_image = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
#     balanced_image[:, :, 0] = clahe.apply(balanced_image[:, :, 0])
#     balanced_image = cv2.cvtColor(balanced_image, cv2.COLOR_LAB2BGR)
#     img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     balanced_image = cv2.equalizeHist(img)
    

#     # plt.imshow(balanced_image)

#     img = balanced_image
#     #         kernel = np.ones((5, 5), np.uint8)  
#     #         cropped_image = cv2.erode(cropped_image, kernel)

#     # init easyocr
#     reader = easyocr.Reader(['vi'], gpu=True)

#     text_boxes = reader.readtext(img)
    
#     def get_center(bbox):
#         bbox = np.array(bbox)
#         x1, y1, x2, y2 = bbox
#         return ([(x1[0] + x2[0]) // 2, (x1[1] + x2[1]) // 2], 
#                 [(y1[0] + y2[0]) // 2, (y1[1] + y2[1]) // 2])

#     # Sort bounding boxes based on the proximity of their center points
#     sorted_text_boxes = sorted(text_boxes, key=lambda box: get_center(box[0]))

#     cropped_images = []

#     threshold = 0.01

#     token_list = []
#     for text_box in sorted_text_boxes:
#         bbox, _, score = text_box

#         if len(bbox) >= 4 and score > threshold:
#             pt1 = (int(bbox[0][0]), int(bbox[0][1]))
#             pt2 = (int(bbox[2][0]), int(bbox[2][1]))

#             cropped_image = img[pt1[1]:pt2[1], pt1[0]:pt2[0]]

#             if len(cropped_image) == 0:
#                 continue

#             ## pytereesact
#             #text = pytesseract_ocr(cropped_image)
#             #print(text)

#             ## vietocr
#             image_pil = Image.fromarray(cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB))
#             image_pil = ImageOps.grayscale(image_pil)
#             s = ocr_detector.predict(image_pil)
#             token_list.append(s)
#     return token_list


# tokens = text_tokens('/kaggle/input/vlsp-data/training-images/training-images/000000000610.jpg')
# print(tokens)