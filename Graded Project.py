#!/usr/bin/env python
# coding: utf-8

# ### Graded Project
# 
# 

#  Machine Learning - Unsupervised Learning

# ### Domain:

# ### ○ E-commerce

# ### Business Context:
# 

# ● Customer segmentation is one of the most important marketing tools at your
# disposal, because it can help a business to better understand its target audience.
# This is because it groups customers based on common characteristics.
# 

# ● Segmentation can be based on the customer’s habits and lifestyle, in
# particular, their buying habits. Different age groups, for example, tend to
# spend their money in different ways, so brands need to be aware of who
# exactly is buying their product.
# 

# ● Segmentation also focuses more on the personality of the consumer,
# including their opinions, interests, reviews, and rating. Breaking down a
# large customer base into more manageable clusters, making it easier to
# identify your target audience and launch campaigns and promote the
# business to the most relevant people
# 

# ### Dataset Description:
# 

# The dataset contains measurements of clothing fit from RentTheRunway.
# RentTheRunWay is a unique platform that allows women to rent clothes for
# various occasions. The collected data is of several categories. This dataset
# contains self-reported fit feedback from customers as well as other side
# information like reviews, ratings, product categories, catalog sizes, customers’
# measurements (etc.)

# ### Data Citation:

# ● Rishabh Misra, Mengting Wan, Julian McAuley "Decomposing Fit Semantics
# for Product Size Recommendation in Metric Spaces". RecSys, 2018.                                                                  
# ● Rishabh Misra, Jigyasa Grover "Sculpting Data for ML: The first act of
# Machine Learning". 2021.
# 

# ### Project Objective:

# Based on the given users and items data of an e-commerce company, segment
# the similar user and items into suitable clusters. Analyze the clusters and provide
# your insights to help the organization promote their business.

# ### ● Import the required libraries :

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.preprocessing import LabelEncoder


# ### 1. Load the data :

# In[2]:


data = pd.read_csv('renttherunways.csv')
data.head()


# ### 2. Check the shape of the data :

# In[3]:


print(data.shape)


# ###  Get information about the data:

# In[4]:


missing_percentage = data.isnull().sum() / len(data) * 100
print(missing_percentage)


# ###  Data cleansing and Exploratory data analysis: 

# ### 3. Check for duplicate records:

# In[5]:


print("Duplicate records:", data.duplicated().sum())


# ### 4. Drop the columns which you think redundant for the analysis.

# In[6]:


data.drop(['user_id', 'review_date'], axis=1, inplace=True)


# ### 5. Check the column 'weight', Is there any presence of string data?            If yes, remove the string data and convert to float. 
# 

# In[7]:


data['weight'] = data['weight'].str.replace('lbs', '').astype(float)


# In[8]:


print(data['weight'])


# ### 6. Check the unique categories for the column 'rented for' and group 'party: cocktail' category with 'party'. 

# In[9]:


data['rented for'] = data['rented for'].replace('party: cocktail', 'party')


# In[10]:


print(data['rented for'])


# ### 7. The column 'height' is in feet with a quotation mark, Convert to inches with float datatype.

# In[11]:


data['height'] = data['height'] * 12


# In[12]:


print(data['height'])


# ### 8. Check for missing values in each column of the dataset? If it exists, impute them with appropriate methods.

# In[13]:


print("Missing values:\n", data.isnull().sum())


# ### 9. Check the statistical summary for the numerical and categorical columns and write your findings.

# In[14]:


numeric_cols = data.select_dtypes(include=np.number)
print(numeric_cols.describe())

categorical_cols = data.select_dtypes(include=object)
print(categorical_cols.describe())


# ### 10. Are there outliers present in the column age? If yes, treat them with the appropriate method.

# In[15]:


q1 = data['age'].quantile(0.25)
q3 = data['age'].quantile(0.75)
iqr = q3 - q1
lower_bound = q1 - 1.5 * iqr
upper_bound = q3 + 1.5 * iqr
data = data[(data['age'] >= lower_bound) & (data['age'] <= upper_bound)]


# In[16]:


print(data['age'])


# ### 11. Check the distribution of the different categories in the column 'rented for' using appropriate plot. 

# In[17]:


sns.countplot(data=data, x='rented for')
plt.xticks(rotation=45)
plt.show()


# ### ● Data Preparation for model building: 

# ### 12. Encode the categorical variables in the dataset.

# In[18]:


from sklearn.preprocessing import LabelEncoder

# Create a copy of the data
data_encoded = data.copy()

# Encode categorical variables using LabelEncoder
encoder = LabelEncoder()
categorical_cols = data.select_dtypes(include='object').columns

for col in categorical_cols:
    data_encoded[col] = encoder.fit_transform(data_encoded[col])

# Fill missing values with the mean
data_encoded.fillna(data_encoded.mean(), inplace=True)


# ### 13. Standardize the data, so that the values are within a particular range.

# In[19]:


scaled = StandardScaler().fit_transform(data_encoded.values)  # Standardize the data
data_scaled = pd.DataFrame(scaled, index=data_encoded.index, columns=data_encoded.columns)


# In[20]:


print(data_scaled[:3])


# ### ● Principal Component Analysis and Clustering:

# ### 14. Apply PCA on the above dataset and determine the number of PCA components to be used so that 90-95% of the variance in data is explained by the same.

# In[21]:


pca = PCA(n_components=0.95)
pca_data = pca.fit_transform(data_scaled)


# In[22]:


cov_matrix = np.cov(data_scaled.T)
cov_matrix


# In[23]:


eig_vals, eig_vectors = np.linalg.eig(cov_matrix)
print('eigin vals:','\n',eig_vals)
print('\n')
print('eigin vectors','\n',eig_vectors)


# In[24]:


total = sum(eig_vals)
var_exp = [(i/total)*100 for i in sorted(eig_vals, reverse=True)]
cum_var_exp = np.cumsum(var_exp)
print('Explained Variance:', var_exp)
print('Cummulative Variance Explained:', cum_var_exp)


# In[25]:


plt.bar(range(len(var_exp)), var_exp, align='center', color='lightgreen', edgecolor='black', label='Explained Variance')
plt.step(range(len(var_exp)), cum_var_exp, where='mid', color='red', label='Cumulative Explained Variance')
plt.xlabel('Principal Components')
plt.ylabel('Explained Variance Ratio')
plt.title('Scree Plot')
plt.legend()
plt.show()


# ### 15. Apply K-means clustering and segment the data. (You may use original data or PCA transformed data)

# ### a. Find the optimal K Value using elbow plot for K Means clustering.

# In[26]:


inertia = []


# In[27]:


k_values = range(2, 11)
for k in k_values:
    # Create a K-means clustering model
    kmeans_model = KMeans(n_clusters=k, random_state=42)
    
    # Fit the model to the data
    kmeans_model.fit(data_scaled)  # Use the original scaled data or pca_data
    
    # Append the inertia value to the list
    inertia.append(kmeans_model.inertia_)


# In[28]:


plt.plot(k_values, inertia, marker='o')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('Inertia')
plt.title('Elbow Plot for K-means Clustering')
plt.show()


# ### b. Build a Kmeans clustering model using the obtained optimal K value from the elbow plot.

# In[29]:


# Choose the optimal K value based on the elbow plot
optimal_k = 4


# In[30]:


# Create a K-means clustering model with the optimal K value
kmeans_model = KMeans(n_clusters=optimal_k, random_state=42)


# In[31]:


# Fit the model to the data
kmeans_model.fit(data_scaled)  # Use the original scaled data or pca_data


# In[32]:


# Get the cluster labels for the data
kmeans_labels = kmeans_model.labels_


# ### c. Compute silhouette score for evaluating the quality of the K Means clustering technique.

# In[33]:


from sklearn.metrics import silhouette_score


# In[34]:


silhouette_score = silhouette_score(data_scaled, kmeans_labels)  # Use the original scaled data or pca_data


# In[35]:


print("Silhouette Score for K-means Clustering:", silhouette_score)


# ### 16. Apply Agglomerative clustering and segment the data. (You may use original data or PCA transformed data)

# ### a. Find the optimal K Value using dendrogram for Agglomerative clustering.

# In[36]:


from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib.pyplot as plt


# In[37]:


sample_size = 1000
sample_data = data_encoded.sample(n=sample_size, random_state=42)


# In[38]:


linkage_matrix = linkage(sample_data, method='ward')


# In[39]:


plt.figure(figsize=(10, 6))
dendrogram(linkage_matrix)
plt.title('Dendrogram for Agglomerative Clustering')
plt.xlabel('Samples')
plt.ylabel('Distance')
plt.show()


# ### b. Build a Agglomerative clustering model using the obtained optimal K value observed from dendrogram.

# In[40]:


from sklearn.cluster import AgglomerativeClustering


# In[41]:


optimal_k = 4


# In[42]:


# Build the Agglomerative clustering model with the optimal K value
agglomerative_model = AgglomerativeClustering(n_clusters=optimal_k, affinity='euclidean', linkage='ward')
agglomerative_model.fit(sample_data)


# In[43]:


# Get the cluster labels for the sample data
agglomerative_labels = agglomerative_model.labels_


# ### c. Compute silhouette score for evaluating the quality of the Agglomerative clustering technique.

# In[44]:


from sklearn.metrics import silhouette_score


# In[48]:


agglomerative_silhouette_score = silhouette_score(sample_data, agglomerative_labels)
print("Silhouette Score for Agglomerative Clustering:", agglomerative_silhouette_score)


# ### ● Conclusion :

# In[55]:


data['kmeans_Cluster'] = kmeans_labels


# In[60]:


agglomerative_labels_shortened = agglomerative_labels[:sample_data.shape[0]]


# In[63]:


sns.boxplot(x='kmeans_Cluster', y='age', data=data)
plt.xlabel('kmeans Cluster')
plt.ylabel('Age')
plt.title('Bivariate Analysis: kmeans Cluster vs Age')
plt.show()

