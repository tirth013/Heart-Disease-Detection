# Importing the libraries
import numpy as np
import pandas as pd
import seaborn as sn

# Importing the dataset
dataset = pd.read_csv(r'C:\Users\MANAN\Desktop\SE project\Model\HealthData.csv')
# dataset = pd.read_csv(r'HealthData.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 13].values

# Handling missing data
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
imputer = imputer.fit(X[:, 11:13])
X[:, 11:13] = imputer.transform(X[:, 11:13])

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Exploring the dataset
sn.countplot(x='num', data=dataset)
dataset.num.value_counts()

# Fitting Decision Tree Classification to the Training set
from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(criterion='entropy', random_state=0)
classifier.fit(X_train, y_train)

# Saving the model
import joblib
filename = 'decision_tree_model.pkl'
joblib.dump(classifier, filename)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Accuracy Score
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy Score:", accuracy)

# Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", cm)

# Classification Report
from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))

# Prediction for new dataset
# Newdataset = pd.read_csv('newdata.csv')

# # Ensure the new dataset has no feature names by converting it to a NumPy array
# X_new = Newdataset.values

# # Make predictions using the trained classifier
# ynew = classifier.predict(X_new)

# print("New Predictions:", ynew)