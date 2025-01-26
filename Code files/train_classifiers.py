import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GroupKFold, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import pickle

# Load the datasets
features_df = pd.read_csv('150features.csv')
metadata_df = pd.read_csv('metadata.csv')

# Extract the image IDs from the features file
image_ids = features_df['image_ids']

# Remove the ".png" extension from the img_id in metadata_df
metadata_df['img_id'] = metadata_df['img_id'].str.replace('.png', '', regex=False)

# Filter the metadata based on these image IDs
filtered_metadata = metadata_df[metadata_df['img_id'].isin(image_ids)]

# Ensure both DataFrames are sorted by image_id to align them
features_df = features_df.set_index('image_ids').loc[filtered_metadata['img_id']].reset_index()
filtered_metadata = filtered_metadata.set_index('img_id').loc[features_df['image_ids']].reset_index()

# Extract the label
label = np.array(filtered_metadata['diagnostic'])

# Define the feature names
ordinal_features = ['Best Asymmetry', 'Mean Asymmetry']
binary_features = ["Red1", "Red2", "White", "Black", "Light brown", "Dark brown", "Blue gray", "Has Veil?"]

# Apply Min-Max scaling to the ordinal features
scaler = MinMaxScaler()
features_df[ordinal_features] = scaler.fit_transform(features_df[ordinal_features])

# Make the dataset
x = np.array(features_df[ordinal_features + binary_features])
y = np.isin(label, ['BCC'])  # True if label is BCC
patient_id = filtered_metadata['patient_id']

# Check if there are any samples
if len(x) == 0 or len(y) == 0:
    raise ValueError("The filtered dataset is empty. Please check the filtering criteria and the input files.")

# Prepare cross-validation - images from the same patient must always stay together
num_folds = 5
group_kfold = GroupKFold(n_splits=num_folds)

# Different classifiers to test out
param_grid = {'n_neighbors': [1, 3, 5, 7, 9]}
knn = KNeighborsClassifier()
grid_search = GridSearchCV(knn, param_grid, cv=group_kfold.split(x, y, patient_id), scoring='accuracy')

# Perform grid search to find the best parameters
grid_search.fit(x, y)
best_knn = grid_search.best_estimator_

# Evaluate using cross-validation
acc_val = np.empty(num_folds)
precision_val = np.empty(num_folds)
recall_val = np.empty(num_folds)
f1_val = np.empty(num_folds)
conf_matrix_sum = np.zeros((2, 2))  # Initialize a confusion matrix sum

for i, (train_index, val_index) in enumerate(group_kfold.split(x, y, patient_id)):
    x_train = x[train_index, :]
    y_train = y[train_index]
    x_val = x[val_index, :]
    y_val = y[val_index]
    
    # Train the classifier
    best_knn.fit(x_train, y_train)
    
    # Evaluate the classifier
    y_pred = best_knn.predict(x_val)
    acc_val[i] = accuracy_score(y_val, y_pred)
    precision_val[i] = precision_score(y_val, y_pred)
    recall_val[i] = recall_score(y_val, y_pred)
    f1_val[i] = f1_score(y_val, y_pred)
    
    # Compute confusion matrix
    conf_matrix = confusion_matrix(y_val, y_pred)
    conf_matrix_sum += conf_matrix  # Accumulate the confusion matrices
    

# Average over all folds
average_acc = np.mean(acc_val)
average_precision = np.mean(precision_val)
average_recall = np.mean(recall_val)
average_f1 = np.mean(f1_val)

print('Average Accuracy: {:.3f}'.format(average_acc))
print('Average Precision: {:.3f}'.format(average_precision))
print('Average Recall: {:.3f}'.format(average_recall))
print('Average F1 Score: {:.3f}'.format(average_f1))

# Print the average confusion matrix
print('Average Confusion Matrix:')
print(conf_matrix_sum / num_folds)

# Train the final model on all data
best_knn.fit(x, y)

# Save the trained classifier
filename = 'groupE_classifier.sav'
pickle.dump(best_knn, open(filename, 'wb'))
