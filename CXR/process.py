from test_crop import generate_mask, generate_bounding_box
import os
from imutils import paths
import cv2

num = len(os.listdir('data/ctest'))



img_rows, img_cols, img_channel = 224, 224, 3


'''
imagePaths = sorted(list(paths.list_images("data/ctest")))

for imagePath in imagePaths:
    # load the image, pre-process it, and store it in the data list
    image = cv2.imread(imagePath)
    image = cv2.resize(image, (img_rows, img_rows))
    cv2.imwrite(imagePath, image)

generate_mask('data/ctest', num, 'data/mask')
'''

generate_bounding_box("data/ctest/", "data/mask/", "bounding_box.csv", "data/crop/")
