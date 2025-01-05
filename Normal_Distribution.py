import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from scipy.stats import norm


#Load the dataSet
url='https://storage.googleapis.com/download.tensorflow.org/data/creditcard.csv'
data=pd.read_csv(url)
data.head()
data.value_counts("Class")
data.drop(columns=['Time'],inplace=True)


#Separate features (X) and target (y)
X = data.drop(columns=['Class','Amount'])
y = data['Class']

X_train,X_test,Y_train,y_test = train_test_split(X, y, test_size=0.3,random_state=42,stratify=y)

print(f"Training set size: {X_train.shape[0]} Test set size: {X_test.shape[0]}")


#Compute the mean and variance for each feature
mean = np. mean(X_train, axis=0)
std_dev = np.std(X_train, axis=0)
print(f"Mean shape: {mean.shape}, Standard deviation shape: {std_dev.shape}")


#Define the Gaussian PDF 
def gaussian_pdf(x, mean, std_dev):
  return (1 / (np.sqrt(2 * np.pi) * std_dev)) * np.exp(-0.5 * ((x - mean) / std_dev)**2)

#Compute probabilities for the test set
probabilities = np.prod(gaussian_pdf(X_test, mean, std_dev),axis=1)


#Visualize the probabilities
plt.figure (figsize=(8, 6))
plt.hist(probabilities, bins=50, alpha=0.7, color='magenta', edgecolor='red')
plt.title("Probability Distribution of Test Data")
plt.xlabel("Probability")
plt.ylabel("Frequency") 


#Define anomaly detection function
def detect_anonalies (probabilities, epsilon) :
  return (probabilities< epsilon).astype(int)

#Set threshold (epsilon) and classify anomalies
epsilon = 1e-20 # Adjust this value to tune the model
y_pred = detect_anonalies(probabilities, epsilon)

#Evaluate performance
print(f"Number of anomalies detected: {np.sum(y_pred)}")


palette=sns.light_palette("magenta",as_cmap=True)

#Plot a heatmap of the confusion matrix
plt.figure(figsize=(8,6))
sns.heatmap(cm,annot=True,fmt="g",cmap=palette,xticklabels=["Honest","Fraudulent"],yticklabels=["Honest","Fraudulent"],cbar=False)
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.title("Confusion Matrix")
plt.show()