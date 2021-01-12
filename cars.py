#K-Means Clustering

# Importation des bibliothèques
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importation de dataset
dataset = pd.read_csv('cars.csv')

X = dataset.iloc[:,:-1].values

X = pd.DataFrame(X)
X = X.convert_objects(convert_numeric=True)
X.columns = ['mpg', ' cylinders', ' cubicinches', ' hp', ' weightlbs', ' time-to-60', 'year']

# Élimination des valeurs nulles
for i in X.columns:
    X[i] = X[i].fillna(int(X[i].mean()))
for i in X.columns:
    print(X[i].isnull().sum())
    

# Utilisation de la méthode du coude pour trouver le nombre optimal de clusters
from sklearn.cluster import KMeans
wcss = []
for i in range(1,11):
    kmeans = KMeans(n_clusters=i,init='k-means++',max_iter=300,n_init=10,random_state=0)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)
plt.plot(range(1,11),wcss)
plt.title('La méthode de coude')
plt.xlabel('Nombre de clusters')
plt.ylabel('WCSS')
plt.show()

# Application de k-means au dataset
kmeans = KMeans(n_clusters=3,init='k-means++',max_iter=200,n_init=10,random_state=0) 
y_kmeans = kmeans.fit_predict(X)

X = X.as_matrix(columns=None)

# Visualiser les clusters
plt.scatter(X[y_kmeans == 0, 0], X[y_kmeans == 0,1],s=100,c='red',label='US')
plt.scatter(X[y_kmeans == 1, 0], X[y_kmeans == 1,1],s=100,c='blue',label='Japan')
plt.scatter(X[y_kmeans == 2, 0], X[y_kmeans == 2,1],s=100,c='green',label='Europe')
plt.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1],s=300,c='yellow',label='Centroids')
plt.title('Clusters de marques automobiles')
plt.legend()
plt.show()