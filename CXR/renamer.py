covid_dir = 'data/covid'
pneumonia_dir = 'data/pneumonia'


for i, filename in enumerate(os.listdir(pneumonia_dir)):
    os.rename(os.path.join(pneumonia_dir, filename), os.path.join(pneumonia_dir, "PNEUMONIA_" + str(i) + ".jpg"))
