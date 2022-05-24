# Fashion_product_recomendation
The purpose of this project is to create an image based recomendation system which recomends the fashion products similar to the one slected by the user 

Dataset used - https://www.kaggle.com/datasets/vikashrajluhaniwal/fashion-images

The project is divided into 2 files 

1st - feature_extraction.py
      The purpose of this file is to extract features from the images corresponding to all the 4 categories available in the dataset
      for extracting features we used ResNet50 and saved the extracted features for all categories in a file to prevent re traing of model every time we
      execute the code 
      (we make seperate files for each category as cross category recomendation is not advisable i.e we cannot suggest a women top to a boy or man )
      
2nd - recomendation_model.py
      This file initially loads all the extracted feature files where each file corresponds to a category in the dataset 
      According to the selected product id of the user it performs following steps:-
      step1 - Finds the gender corresponding to the given product id
      Step2 - Based on gender it loads the extracted features file that corresponds to that particular gender.
      Step3 - Next it calculates the eucledian distance between the feature matrix of given product and all other products in the set and returs the top5 
              productas with least euclidean distance resulting in similar products.
              
           
      
 
