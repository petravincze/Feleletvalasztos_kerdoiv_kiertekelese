# szukseges csomagok importalasa
import cv2


class Shape:
    def __init__(self):
        pass
    def detect(self, c):
        
        # initialize the shape name and approximate the contour
        shape = "unidentified"
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.04 * peri, True)
        
        if len(approx) == 4:
            shape = "negyzet"
        else:
            shape = "kor"

        return shape