import os
import shutil
import random
from data_augmentor import augment_pneumonia, add_noise_all_training

#covid sources are 
#china = 393
#velancia = 377
#grenada = 424
#tcia = 16
#germany 135

covid_sources = ["china", "velancia", "granada", "tcia", "germany"]
unseen_covid = ["velancia"]

#pneumonia sources are
#padchest = 289
#NIH = 271
#chexpert = 32

pneumonia_sources = ["padchest", "NIH", "chexpert"]
unseen_pnuemonia = ["NIH"]

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
    if source in unseen_covid:
        continue

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
    







#add the unseen sources
for source in unseen_covid:
    print("Moving from unseen source: {}".format(source))

    main = os.listdir(processed_covid + "/" + source)

    for file in main:
        shutil.copyfile(processed_covid + "/" + source + "/" + file, test_unseen_path + "/covid/" + file)


for source in unseen_pnuemonia:
    print("Moving from unseen source: {}".format(source))

    main = os.listdir(processed_pneumonia + "/" + source)

    for file in main:
        shutil.copyfile(processed_pneumonia + "/" + source + "/" + file, test_unseen_path + "/pneumonia/" + file)







#optional
#augment training pneumonia data
#augment_pneumonia()






#balance train folder
for f in [['data/train/covid', 'data/train/pneumonia'], ['data/valid/covid', 'data/valid/pneumonia'], ['data/test-seen/covid', 'data/test-seen/pneumonia'], ['data/test-unseen/covid', 'data/test-unseen/pneumonia']]:

    t_covid = os.listdir(f[0])
    t_pneumonia = os.listdir(f[1])

    if len(t_covid) > len(t_pneumonia):
        diff = len(t_covid) - len(t_pneumonia)

        for i in range(diff):
            file = random.choice(os.listdir(f[0]))
            os.remove(f[0]+"/"+file)



#optional
#add gauss noise to all training data
#add_noise_all_training()