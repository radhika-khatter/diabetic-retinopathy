"""Step 2: Retina crop, resize, CLAHE. Returns RGB uint8."""
import cv2, numpy as np

def crop_fundus(img, tol=7):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ys, xs = np.where(gray > tol)
    if len(ys) == 0: return img
    return img[ys.min():ys.max() + 1, xs.min():xs.max() + 1]

def clahe_bgr(img):
    l, a, b = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2LAB))
    l = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(l)
    return cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)

def preprocess(img_bgr, size=224):
    img = cv2.resize(crop_fundus(img_bgr), (size, size))
    return cv2.cvtColor(clahe_bgr(img), cv2.COLOR_BGR2RGB)
