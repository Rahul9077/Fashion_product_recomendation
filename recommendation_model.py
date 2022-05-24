import numpy as np
import pandas as pd
from sklearn.metrics import pairwise_distances
from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

from IPython.display import display, Image



fashion_df = pd.read_csv("Dataset/data/fashion.csv")
boys_extracted_features = np.load('Dataset/data/Boys_ResNet_features.npy')
boys_Productids = np.load('Dataset/data/Boys_ResNet_feature_product_ids.npy')
girls_extracted_features = np.load('Dataset/data/Girls_ResNet_features.npy')
girls_Productids = np.load('Dataset/data/Girls_ResNet_feature_product_ids.npy')
men_extracted_features = np.load('Dataset/data/Men_ResNet_features.npy')
men_Productids = np.load('Dataset/data/Men_ResNet_feature_product_ids.npy')
women_extracted_features = np.load('Dataset/data/Women_ResNet_features.npy')
women_Productids = np.load('Dataset/data/Women_ResNet_feature_product_ids.npy')
fashion_df["ProductId"] = fashion_df["ProductId"].astype(str)



def get_similar_products_cnn(product_id, num_results):
    """these if, elif staements will chk weather the product id entered by the user belongs to boys ,girls, men or women
        eg - let us suppose the entered product id is 12345
        so,
        fashion_df[fashion_df['ProductId']=='21838']
        the above line will find the row from fashion_df corresponding to given product id

        ProductId	Gender	Category	SubCategory	ProductType	Colour	Usage	ProductTitle	                                          Image	        ImageURL
        21838	    Boys	Apparel	    Topwear	    Tshirts	    Yellow	Casual	Chhota Bheem Kids Boys Bheem Expressions Yellow T-shirt	  21838.jpg	    http://assets.myntassets.com/v1/images/style/properties/2efd395f86e19cadd0bd57f45a51adbd_images.jpg

        now rest part will extract the gender from the above row
        Gender- Boys

        based on the extracted gender it will load the corresponding set of extracted features and product ids
    """
    extracted_features = 0
    Productids = 0

    if(fashion_df[fashion_df['ProductId']==product_id]['Gender'].values[0]=="Boys"):
        extracted_features = boys_extracted_features
        Productids = boys_Productids
    elif(fashion_df[fashion_df['ProductId']==product_id]['Gender'].values[0]=="Girls"):
        extracted_features = girls_extracted_features
        Productids = girls_Productids
    elif(fashion_df[fashion_df['ProductId']==product_id]['Gender'].values[0]=="Men"):
        extracted_features = men_extracted_features
        Productids = men_Productids
    elif(fashion_df[fashion_df['ProductId']==product_id]['Gender'].values[0]=="Women"):
        extracted_features = women_extracted_features
        Productids = women_Productids

    Productids = list(Productids)
    doc_id = Productids.index(product_id)
    pairwise_dist = pairwise_distances(extracted_features, extracted_features[doc_id].reshape(1,-1))

    indices = np.argsort(pairwise_dist.flatten())[0:num_results]
    pdists  = np.sort(pairwise_dist.flatten())[0:num_results]


    ip_row = fashion_df[['ImageURL','ProductTitle']].loc[fashion_df['ProductId']==Productids[indices[0]]]

    for indx, row in ip_row.iterrows():
        display(Image(url=row['ImageURL'], width = 224, height = 224,embed=True))
        print('Product Title: ', row['ProductTitle'])


    for i in range(1,len(indices)):
        rows = fashion_df[['ImageURL','ProductTitle']].loc[fashion_df['ProductId']==Productids[indices[i]]]

        for indx, row in rows.iterrows():
            display(Image(url=row['ImageURL'], width=224, height=224, embed=True))
            print('Product Title: ', row['ProductTitle'])
            print('Euclidean Distance from input image:', pdists[i])



product_id = input("Enter the item id")
number_of_products = input("Enter number of products to be recommended")

get_similar_products_cnn(product_id=product_id,num_results=number_of_products)
