from threading import local
import cv2
import urllib.request as urllib
import numpy as np

from urllib.request import Request, urlopen


class ImageLoader():
    def __init__(self, id : str = None, img: np.array = None):
        """
        id should be in ' ' not in " " 
        """
        self.id = id
        self.img = img
        
        if self.id is not None:
            self.url = f"https://www.missionjuno.swri.edu/Vault/VaultOutput?VaultID={id}"
        
    def load(self):
        if self.id is not None:
            req = Request(
            url= self.url, 
            headers={'User-Agent': 'Mozilla/5.0'}
            )
            webpage = urlopen(req).read()
            arr = np.asarray(bytearray(webpage), dtype=np.uint8)
            img = cv2.imdecode(arr, -1) # 'Load it as it is'
            return img
        elif self.img is not None:
            return self.img
