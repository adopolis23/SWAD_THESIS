import matplotlib.pyplot as plt
import numpy as np
import random
import cv2
import os

#data sources
pneumonia_sources = ["padchest", "NIH", "chexpert"]
held_out = ["NIH"]

pneumonia_training_dir = "data/train/pneumonia/"

#percent of total pneumonia images to be augmented for now we do all of them to double dataset size
#CANNOT be greater than 1 bad things will happen
percent_aug = 1.0

#percent of images that get gaussian noise
img_percent_noise = 1.0

#for gaussian noise
mean = 0
variance = 10




#for testing
'''
if os.path.isdir('data/augmented_pneumonia') is False:
    os.makedirs('data/augmented_pneumonia')


#for each source count the files
file_count = 0
for source in pneumonia_sources:
    file_count += len(os.listdir("data/processed_pneumonia/"+source))

print("Total Files Is: {}".format(file_count))
'''





#input: ndarray of image
#output: ndarray of image with added gaussian noise
def add_gauss_noise(image, var, mean):
    row,col,ch= image.shape
    sigma = var**0.5
    gauss = np.random.normal(mean,sigma,(row,col,ch))
    gauss = gauss.reshape(row,col,ch)
    noisy = image + gauss
    return noisy

#input: image
#output: image flipped over the x-axis
def flip_horizontally(image):
    new_image = cv2.flip(image, 1)
    return new_image


#looks in the training directory for pnuemonia and augments them and adds to same directory. 
def augment_pneumonia():

    print("Augmenting Files")

    files = os.listdir(pneumonia_training_dir)

    num_current_files = len(files)
    print("Total Files Is: {}".format(num_current_files))

    num_files_to_augmnet = num_current_files * percent_aug
    print("Creating {} augmented files.".format(int(num_files_to_augmnet)))




    while num_files_to_augmnet > 0:
        file = random.choice(files)
        files.remove(file)

        #load image and convert to ndarray
        image = cv2.imread(pneumonia_training_dir + file)
        image=np.array(image)

        #apply augments
        new_image = flip_horizontally(image)
        new_image = add_gauss_noise(new_image, variance, mean)

        #save image
        cv2.imwrite(pneumonia_training_dir + "/AUGMENT" + file, new_image)

        num_files_to_augmnet -= 1











