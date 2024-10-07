from sklearn.impute import SimpleImputer
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
import time

# Load the dataset (adult.data)
column_names = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 
                'occupation', 'relationship', 'race', 'sex', 'capital-gain', 
                'capital-loss', 'hours-per-week', 'native-country', 'income']

# Load the full data
adult_data = pd.read_csv('adult.data', delimiter=",", header=None, names=column_names, na_values=' ?')

# Impute missing values using SimpleImputer
imputer = SimpleImputer(strategy='most_frequent')

# Fill missing values in the dataset
adult_data_imputed = pd.DataFrame(imputer.fit_transform(adult_data), columns=adult_data.columns)

# Scale numerical columns **before** one-hot encoding
numerical_columns = ['age', 'fnlwgt', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week']
scaler = StandardScaler()
adult_data_imputed[numerical_columns] = scaler.fit_transform(adult_data_imputed[numerical_columns])

# One-hot encode categorical variables
adult_data_encoded = pd.get_dummies(adult_data_imputed, drop_first=True)

# Convert the 'income' column to integers
adult_data_encoded['income_ >50K'] = adult_data_encoded['income_ >50K'].astype(int)

# Split features (X) and target (y)
X = adult_data_encoded.drop('income_ >50K', axis=1)
y = adult_data_encoded['income_ >50K']

# Stratified train-test split to ensure both classes are represented
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# ====================== Decision Tree ======================

# Initialize and train Decision Tree model with class weights to handle imbalance
#tree = DecisionTreeClassifier(random_state=42, class_weight='balanced', criterion= 'gini')
tree = DecisionTreeClassifier(random_state=42, class_weight='balanced', criterion= 'entropy')


# Measure training time for Decision Tree
start_time = time.time()
tree.fit(X_train, y_train)
tree_train_time = time.time() - start_time

# Predict on the test data
y_pred_tree = tree.predict(X_test)

# Evaluate Decision Tree
tree_accuracy = accuracy_score(y_test, y_pred_tree)
tree_class_report = classification_report(y_test, y_pred_tree)

# Output results for Decision Tree
print(f"\nDecision Tree Accuracy: {tree_accuracy}")
print(f"Decision Tree Training Time: {tree_train_time} seconds")
print(f"Decision Tree Classification Report:\n{tree_class_report}")

# ====================== K-Nearest Neighbors (KNN) ======================

# Initialize and train KNN model (with k=5)
knn = KNeighborsClassifier(n_neighbors=5)

# Measure training time for KNN
start_time = time.time()
knn.fit(X_train, y_train)
knn_train_time = time.time() - start_time

# Predict on the test data
y_pred_knn = knn.predict(X_test)

# Evaluate KNN
knn_accuracy = accuracy_score(y_test, y_pred_knn)
knn_class_report = classification_report(y_test, y_pred_knn)

# Output results for KNN
print(f"\nKNN Accuracy: {knn_accuracy}")
print(f"KNN Training Time: {knn_train_time} seconds")
print(f"KNN Classification Report:\n{knn_class_report}")

# ====================== Comparison of Both Models ======================

# Compare Decision Tree and KNN results
print("\n====================== Model Comparison ======================")
print(f"Decision Tree - Accuracy: {tree_accuracy}, Training Time: {tree_train_time} seconds")
print(f"KNN - Accuracy: {knn_accuracy}, Training Time: {knn_train_time} seconds")
