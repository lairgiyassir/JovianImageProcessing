import numpy as np 
import cv2 as cv 
from ..preprocessing import ImageProcessor
import json 

class Utils():

    def __init__(self) -> None:
        pass

    def decode(self, img_coded):
        arr = np.asarray(bytearray(img_coded), dtype=np.uint8)
        img = cv.imdecode(arr, -1) # 'Load it as it is'

        return img
    
    def processing_pipeline(self, img : np.array, jsn):
        list_of_processing = []

        functions = {"enhance_small_scale_clouds": ImageProcessor.enhance_small_scale_clouds,
        "enhance_large_scale_clouds": ImageProcessor.enhance_large_scale_clouds,
        "denoising" : ImageProcessor.denoise, 
        "enhance_contrast" : ImageProcessor.enhance_contrast, 
        "enhance_brightness": ImageProcessor.enhance_brightness,
        "gamma_correction" : ImageProcessor.gamma_corrector,
        "gray_scale" : ImageProcessor.gray_scale,
        }

        for func in jsn:
            if jsn[func]:
                list_of_processing.append(func)

        img_result = functions[list_of_processing[0]](img)


        for i in range(1, len(list_of_processing)):
            img_result = functions[list_of_processing[i]](img_result)
        return img_result

