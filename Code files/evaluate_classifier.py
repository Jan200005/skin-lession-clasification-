import pickle
from extract_features import extract_features
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
from PIL import Image
#Insert the path to your image and your mask. 
#If the mask is a png file, use the png_to_numpy_array function

image_path=""
mask_path=""

def png_to_numpy_array(file_path):
    # Open the image file
    image = Image.open(file_path).convert('L')  # convert to grayscale
    # Convert the image to a numpy array
    numpy_array = np.array(image)
    return numpy_array

image = plt.imread(image_path)
image_for_veil = cv2.imread(image_path, cv2.IMREAD_COLOR)
mask = np.load(mask_path)
#mask = png_to_numpy_array(mask_path)



# The function that should classify new images.
# The image and mask are the same size, and are already loaded using plt.imread
def classify(img, mask, img_for_veil):
    # Extract features (the same ones that you used for training)
    x = extract_features(img, mask, img_for_veil)
    
    # Ensure the features are in a DataFrame for consistency
    feature_names = ['Best Asymmetry', 'Mean Asymmetry', "Red1", "Red2", "White", "Black", "Light brown", "Dark brown", "Blue gray", "Has Veil?"]
    features_df = pd.DataFrame([x], columns=feature_names)
    
    # Separate ordinal and binary features
    ordinal_features = ['Best Asymmetry', 'Mean Asymmetry']
    binary_features = ["Red1", "Red2", "White", "Black", "Light brown", "Dark brown", "Blue gray", "Has Veil?"]
    
    # Apply Min-Max scaling to the ordinal features
    scaler = MinMaxScaler()
    features_df[ordinal_features] = scaler.fit_transform(features_df[ordinal_features])
    
    # Combine the features into a single array
    x_processed = np.array(features_df[ordinal_features + binary_features])
    
    # Load the trained classifier
    classifier = pickle.load(open('Data/groupE_classifier.sav', 'rb'))
    
    # Use it on this example to predict the label AND posterior probability
    pred_label = classifier.predict(x_processed)
    pred_prob = classifier.predict_proba(x_processed)
    
    return pred_label, pred_prob

