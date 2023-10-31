import os
import cv2
import numpy as np

data_folder = "data/"
domains = ["art_painting/", "cartoon/", "photo/", "sketch/"]
classes = ["dog/", "elephant/", "giraffe/", "guitar/", "horse/", "house/", "person/"]

for domain in domains:
        print("Processing domain: {}".format(domain))

        for _class in classes:
            
            files = os.listdir(data_folder + domain + _class)
                

            for file in files:

                image = cv2.imread(data_folder + domain + _class + file)
                image=cv2.resize(image, (224, 224), interpolation = cv2.INTER_AREA)
                #image=np.array(image)

                cv2.imwrite(data_folder + domain + _class + file, image)
                


                

