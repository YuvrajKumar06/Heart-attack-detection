import cv2
import numpy as np
import os

def preprocess_ecg(image_path, output_folder="temp/processed"):

    os.makedirs(output_folder, exist_ok=True)

    img = cv2.imread(image_path)
    if img is None:
        raise ValueError("Image not found")

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    lower_red1 = np.array([0, 20, 50]) 
    upper_red1 = np.array([15, 255, 255])
    lower_red2 = np.array([160, 20, 50])
    upper_red2 = np.array([180, 255, 255])

    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    red_mask = cv2.add(mask1, mask2)

    img[red_mask > 0] = (255, 255, 255)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)

    kernel = np.ones((2, 2), np.uint8)
    cleaned = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

    output_path = os.path.join(output_folder, "processed.jpg")
    cv2.imwrite(output_path, cleaned)

    return output_path