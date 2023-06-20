import os
import shutil
import random

#covid sources are 
#china
#velancia
#grenada

covid_sources = ["china", "velancia", "granada"]
unseen_covid = []

#pneumonia sources are
#padchest
#NIH
#chexpert

pneumonia_sources = ["padchest", "NIH", "chexpert"]

test_split = 0.1
val_split = 0.1


if os.path.isdir('data/train/covid') is False:
    os.makedirs('data/train/covid')
    os.makedirs('data/train/pneumonia')
    os.makedirs('data/valid/covid')
    os.makedirs('data/valid/pneumonia')
    os.makedirs('data/test-seen/covid')
    os.makedirs('data/test-seen/pneumonia')
    os.makedirs('data/test-unseen/covid')
    os.makedirs('data/test-unseen/pneumonia')

processed_covid = "data/processed_covid"
processed_pneumonia = "data/processed_pneumonia"

train_path = "data/train"
valid_path = "data/valid"
test_seen_path = "data/test-seen"
test_unseen_path = "data/test-unseen"




#split and move all covid sources
for source in covid_sources:

    print("Moving from source: {}".format(source))

    #split and move the 'china' data into train, val, and test-seen
    main = os.listdir("data/processed_covid/" + source)
    test_number = int(len(main) * test_split)
    val_number = int(len(main) * val_split)

    #copy all files into train
    for file in main:
        shutil.copyfile(processed_covid+"/"+ source + "/"+file, train_path+"/covid/"+file)


    #move files into test_seen
    for i in range(test_number):
        file = random.choice(os.listdir(train_path+"/covid/"))
        shutil.move(train_path+"/covid/"+file, test_seen_path+"/covid/"+file)

    #move files into val
    for i in range(val_number):
        file = random.choice(os.listdir(train_path+"/covid/"))
        shutil.move(train_path+"/covid/"+file, valid_path+"/covid/"+file)
    

#split and move all pneumonia sources
for source in pneumonia_sources:

    print("Moving from source: {}".format(source))

    #split and move the 'china' data into train, val, and test-seen
    main = os.listdir("data/processed_pneumonia/" + source)
    test_number = int(len(main) * test_split)
    val_number = int(len(main) * val_split)

    #copy all files into train
    for file in main:
        shutil.copyfile(processed_pneumonia+"/"+ source + "/"+file, train_path+"/pneumonia/"+file)


    #move files into test_seen
    for i in range(test_number):
        file = random.choice(os.listdir(train_path+"/pneumonia/"))
        shutil.move(train_path+"/pneumonia/"+file, test_seen_path+"/pneumonia/"+file)

    #move files into val
    for i in range(val_number):
        file = random.choice(os.listdir(train_path+"/pneumonia/"))
        shutil.move(train_path+"/pneumonia/"+file, valid_path+"/pneumonia/"+file)
    