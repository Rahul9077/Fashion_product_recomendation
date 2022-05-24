import numpy as np
import pandas as pd
from keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import pairwise_distances
import requests
from PIL import Image
import pickle
from tensorflow import keras
from datetime import datetime
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from tensorflow.keras.applications.resnet50 import ResNet50
from IPython.display import display, Image



fashion_df = pd.read_csv("Dataset/data/fashion.csv")

fashion_df.head()

apparel_boys = fashion_df[fashion_df["Gender"]=="Boys"]
apparel_girls = fashion_df[fashion_df["Gender"]=="Girls"]
footwear_men = fashion_df[fashion_df["Gender"]=="Men"]
footwear_women = fashion_df[fashion_df["Gender"]=="Women"]

img_width, img_height = 224, 224

# top_model_weights_path = 'resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5'

"""while doing exploratory data analysis we infer the total entries in all 4 categories i.e
    men - 811
    Women - 769
    Boys - 759
    Girls - 567
"""


#for men

train_data_dir = "Dataset/data/Footwear/Men/Images/"

nb_train_samples = 811
epochs = 50
batch_size = 1


def extract_features_men():
    Itemcodes = []
    datagen = ImageDataGenerator(rescale=1. / 255)
    model = ResNet50(include_top=False, weights='imagenet')
    generator = datagen.flow_from_directory(
        train_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode=None,
        shuffle=False)
    for i in generator.filenames:
        Itemcodes.append(i[(i.find("/") + 1):i.find(".")])
    extracted_features = model.predict_generator(generator, nb_train_samples // batch_size)
    extracted_features = extracted_features.reshape((811, 100352))

    np.save(open('Dataset/data/Men_ResNet_features.npy', 'wb'), extracted_features)
    np.save(open('Dataset/data/Men_ResNet_feature_product_ids.npy', 'wb'),
            np.array(Itemcodes))

# for Women

train_data_dir_women = "Dataset/data/Footwear/Women/Images/"

nb_train_samples_women = 769
epochs_women = 50
batch_size_women = 1

def extract_features_women():
    Itemcodes = []
    datagen = ImageDataGenerator(rescale=1. / 255)
    model = ResNet50(include_top=False, weights='imagenet')
    generator = datagen.flow_from_directory(
        train_data_dir_women,
        target_size=(img_width, img_height),
        batch_size=batch_size_women,
        class_mode=None,
        shuffle=False)
    for i in generator.filenames:
        Itemcodes.append(i[(i.find("/") + 1):i.find(".")])
    extracted_features = model.predict_generator(generator, nb_train_samples_women // batch_size_women)
    extracted_features = extracted_features.reshape((769, 100352))

    np.save(open('Dataset/data/Women_ResNet_features.npy', 'wb'), extracted_features)
    np.save(open('Dataset/data/Women_ResNet_feature_product_ids.npy', 'wb'),
            np.array(Itemcodes))

#for Boys
train_data_dir_Boys = "Dataset/data/Apparel/Boys/Images"

nb_train_samples_Boys = 759
epochs_Boys = 50
batch_size_Boys = 1

def extract_features_Boys():
    Itemcodes = []
    datagen = ImageDataGenerator(rescale=1. / 255)
    model = ResNet50(include_top=False, weights='imagenet')
    generator = datagen.flow_from_directory(
        train_data_dir_Boys,
        target_size=(img_width, img_height),
        batch_size=batch_size_Boys,
        class_mode=None,
        shuffle=False)
    for i in generator.filenames:
        Itemcodes.append(i[(i.find("/") + 1):i.find(".")])
    extracted_features = model.predict_generator(generator, nb_train_samples_Boys // batch_size_Boys)
    extracted_features = extracted_features.reshape((759, 100352))


    np.save(open('Dataset/data/Boys_ResNet_features.npy', 'wb'), extracted_features)
    np.save(open('Dataset/data/Boys_ResNet_feature_product_ids.npy', 'wb'),
            np.array(Itemcodes))


# for girls
train_data_dir_Girls = "Dataset/data/Apparel/Girls/Images"

nb_train_samples_Girls = 567
epochs_Girls = 50
batch_size_Girls = 1


def extract_features_Girls():
    Itemcodes = []
    datagen = ImageDataGenerator(rescale=1. / 255)
    model = ResNet50(include_top=False, weights='imagenet')
    generator = datagen.flow_from_directory(
        train_data_dir_Girls,
        target_size=(img_width, img_height),
        batch_size=batch_size_Girls,
        class_mode=None,
        shuffle=False)
    for i in generator.filenames:
        Itemcodes.append(i[(i.find("/") + 1):i.find(".")])
    extracted_features = model.predict_generator(generator, nb_train_samples_Girls // batch_size_Girls)
    extracted_features = extracted_features.reshape((811, 100352))

    np.save(open('/home/ec2-user/Desktop/imp_data/recomend/Men_ResNet_features.npy', 'wb'), extracted_features)
    np.save(open('/home/ec2-user/Desktop/imp_data/recomend/Men_ResNet_feature_product_ids.npy', 'wb'),
            np.array(Itemcodes))

a = datetime.now()

extract_features_men()
extract_features_women()
extract_features_Boys()
extract_features_Girls()
print("Time taken in feature extraction", datetime.now() - a)
