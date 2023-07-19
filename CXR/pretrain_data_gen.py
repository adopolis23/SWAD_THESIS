
import os
import shutil
import random

#normal sources are 


normal_sources = ["China", "NIH", "chexpert", "GRANADA"]
unseen_normal = ["NIH"]


#pneumonia sources are
#padchest = 289
#NIH = 271
#chexpert = 32

pneumonia_sources = ["padchest", "NIH", "chexpert"]
unseen_pnuemonia = ["NIH"]

test_split = 0.1
val_split = 0.1



if os.path.isdir('data/train2/normal') is False:
    os.makedirs('data/train2/normal')
    os.makedirs('data/train2/pneumonia')
    os.makedirs('data/valid2/normal')
    os.makedirs('data/valid2/pneumonia')
    os.makedirs('data/test-seen2/normal')
    os.makedirs('data/test-seen2/pneumonia')

processed_normal = "data/processed_normal"
processed_pneumonia = "data/processed_pneumonia"


train_path = "data/train2"
valid_path = "data/valid2"
test_seen_path = "data/test-seen2"



#split and move all covid sources
for source in normal_sources:
    if source in unseen_normal:
        continue

    print("Moving from source: {}".format(source))

    main = os.listdir("data/processed_normal/" + source)
    test_number = int(len(main) * test_split)
    val_number = int(len(main) * val_split)

    #copy all files into train
    for file in main:
        shutil.copyfile(processed_normal+"/"+ source + "/"+file, train_path+"/normal/"+file)


    #move files into test_seen
    for i in range(test_number):
        file = random.choice(os.listdir(train_path+"/normal/"))
        shutil.move(train_path+"/normal/"+file, test_seen_path+"/normal/"+file)

    #move files into val
    for i in range(val_number):
        file = random.choice(os.listdir(train_path+"/normal/"))
        shutil.move(train_path+"/normal/"+file, valid_path+"/normal/"+file)




#split and move all pneumonia sources
for source in pneumonia_sources:
    if source in unseen_pnuemonia:
        continue

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