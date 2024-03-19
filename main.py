#!/usr/bin/python
# -*- coding: utf-8 -*-
from pycnn import PyCNN
import cv2


def main():
    # Initialize object
    cnn = PyCNN()

    input_img = "images/" + input("Name image: ").strip()

    output_img = 'images/output.png'

    try:
        # Perform respective image processing techniques on the given image
        cnn.cornerDetection(input_img, output_img)
    except Exception as f:
        print(f"File {input_img} does not exist. Error:", f)
        return

    # Uploading images
    img1 = cv2.imread(output_img, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(input_img)

    # Adaptive binarization of the first image
    thresh = cv2.adaptiveThreshold(img1, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 11, 10)

    # Contour search
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Creating a copy of the second image
    overlay = img2.copy()

    # Determining the zoom factor for the radius of the circles
    fixed_radius = int(5 * min(img2.shape[0], img2.shape[1]) / 256)

    # The outline of the black elements in a circle in the second image
    for contour in contours:
        (x, y), _ = cv2.minEnclosingCircle(contour)
        center = (int(x), int(y))
        cv2.circle(overlay, center, fixed_radius, (0, 0, 255), 3)

    # We overlay the processed image on top of the second one
    result = cv2.addWeighted(overlay, 0.5, img2, 0.5, 0)

    # Displaying the result
    cv2.imshow('Result', result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
