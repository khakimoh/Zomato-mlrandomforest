#!/usr/bin/env python
# coding: utf-8

# # Analysis Data of Rating Zomato Resturant 
# ### In this project, we intend to analyze the data of Zomato restaurant and review their ranking method 
# 
# 
# ### Table content: The steps will generally be as follows:
# - Extract and convert data to standard CSV
# - Normalize many data
# - The information is divided into two segment
# - Clustering feature with k-means method unsupervised
# - Merging clusters and sticking the label to data about delivery and classy ambiance
# - Classification data for predicting user ratting
# - Analysis functionality method be run
# 
# 
# ##### Developed by: M.khaki
# 
# #### Data from: kaggle.com
# 

# In[1]:


import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import json
import csv
import os
from sklearn import svm
from sklearn import preprocessing
from sklearn.cluster import KMeans
from sklearn.metrics import RocCurveDisplay
from sklearn.metrics import cohen_kappa_score
from sklearn.model_selection import train_test_split
from collections import Counter

itemNumberFile=[75,75,75,75,68]


# ## Extracte and Convert Data to Standard CSV
# In this block, the required for this project is extracted five JSON data from the zomato restaurant on the Kaggle website.
# Also, these items have been converted from JSON to CSV for further analysis

# In[2]:


def selection_json(values,iterate):
    for i in range(iterate):
        for value in values[i]["restaurants"]:
            item = {
                "id": value["restaurant"]["R"]["res_id"],
                "has_online_delivery":value["restaurant"]["has_online_delivery"],
                "has_table_booking":value["restaurant"]["has_table_booking"],
                "is_delivering_now":value["restaurant"]["is_delivering_now"],
                "switch_to_order_menu":value["restaurant"]["switch_to_order_menu"],
                "cuisines":len(value["restaurant"]["cuisines"].split(",")),
                "average_cost_for_two":value["restaurant"]["average_cost_for_two"],
                "price_range":value["restaurant"]["price_range"],
                "votes":value["restaurant"]["user_rating"]["votes"],
                "aggregate_rating":value["restaurant"]["user_rating"]["aggregate_rating"],
                }
            data.append(item)

data=[]
jsonFile = open('data/file1.json')
values = json.load(jsonFile)
jsonFile.close()
selection_json(values,itemNumberFile[0])
jsonFile = open('data/file2.json')
values = json.load(jsonFile)
jsonFile.close()
selection_json(values,itemNumberFile[1])
jsonFile = open('data/file3.json')
values = json.load(jsonFile)
jsonFile.close()
selection_json(values,itemNumberFile[2])
jsonFile = open('data/file4.json')
values = json.load(jsonFile)
jsonFile.close()
selection_json(values,itemNumberFile[3])
jsonFile = open('data/file5.json')
values = json.load(jsonFile)
jsonFile.close()
selection_json(values,itemNumberFile[4])
jsonData=json.dumps(data)
df = pd.read_json(jsonData)
df.to_csv('zomato_df.csv')
df.describe()


# Change vote to vote/max

# In[3]:


vote_max=df["votes"].max()
df["votes"]=df["votes"]/vote_max


# ## Normalize items of classification2
# cuisines, average cost for two, price range and votes should be normalized between 0 and 1

# In[4]:


df.columns


# Transformation some data from pandas to numpy for standard use in scikit learn

# In[5]:


c2=df[["cuisines","average_cost_for_two","price_range","votes"]].values
df=df[["id", "has_online_delivery", "has_table_booking", "is_delivering_now","switch_to_order_menu","aggregate_rating"]]


# Normalize data with min-max scaler 

# In[6]:


scaler = preprocessing.MinMaxScaler(feature_range=(0, 1)).fit(c2)
c2=scaler.transform(c2)
c2 = pd.DataFrame(data=c2 , columns=["cuisines","average_cost_for_two","price_range","votes"])


# Concat class1 and normalized class2 & Ronded aggregate_rating parameter

# In[7]:


data = pd.concat([df, c2], axis=1)
data.loc[(data["aggregate_rating"] >= 4.5), 'aggregate_rating'] = 5
data.loc[(data["aggregate_rating"] >= 3.8) & (data["aggregate_rating"] < 4.5), 'aggregate_rating'] = 4
data.loc[(data["aggregate_rating"] >= 2.8) & (data["aggregate_rating"] < 3.8), 'aggregate_rating'] = 3
data.loc[(data["aggregate_rating"] >= 1.8) & (data["aggregate_rating"] < 2.8), 'aggregate_rating'] = 2
data.loc[(data["aggregate_rating"] < 1.8), 'aggregate_rating'] = 1
data["aggregate_rating"]=data["aggregate_rating"].astype(int)
data.head()


# In[8]:


rating_histogram = Counter(data["aggregate_rating"])
plt.bar(rating_histogram.keys(), rating_histogram.values())
plt.show()


# # Over sampling
# oversampling helps to data balance on user rating

# In[9]:


from imblearn.over_sampling import SMOTE
oversampel_x=data[["id", "has_online_delivery", "has_table_booking", "is_delivering_now","switch_to_order_menu","cuisines","average_cost_for_two","price_range","votes"]].values
oversampel_y=data[["aggregate_rating"]].values

ratingCount = []
temp_key = []
for i in range(1,6):  
    ratingCount.append(data.loc[(data["aggregate_rating"] == i )].count().aggregate_rating)
max_key = ratingCount.index(np.max(ratingCount))+1
min_key = ratingCount.index(np.min(ratingCount))+1
if(ratingCount[max_key-1]*0.55 > ratingCount[min_key-1]):
    max_value = ratingCount[max_key-1]
    min_value = (max_value*0.45).astype(int)
else:
    max_value = ratingCount[max_key-1]
    min_value=ratingCount[min_key-1]
    
temp_value = []
temp_grow = []   
for i in range(5):
    if(i!= max_key-1 and i!= min_key-1):
        temp_key.append(i+1) 
for i in range(3):
    temp_grow.append(ratingCount[temp_key[i]-1]/ratingCount[min_key-1])
temp_grow_sum = sum(temp_grow)
for i in range(3):
    temp_grow[i]=temp_grow[i]/temp_grow_sum
for i in range(3):
    temp_value.append(((max_value - min_value)*(temp_grow[i]) + min_value).astype(int)) 
strategy = {
    max_key: max_value,
    min_key: min_value,
    temp_key[0]: temp_value[0],
    temp_key[1]: temp_value[1],
    temp_key[2]: temp_value[2],
}

oversample = SMOTE(sampling_strategy=strategy)
oversampel_x, oversampel_y = oversample.fit_resample(oversampel_x, oversampel_y)
counter = Counter(oversampel_y)
oversampel_x = pd.DataFrame(data=oversampel_x , columns=["id", "has_online_delivery", "has_table_booking", "is_delivering_now","switch_to_order_menu","cuisines","average_cost_for_two","price_range","votes"])
oversampel_y = pd.DataFrame(data=oversampel_y , columns=["aggregate_rating"])
data = pd.concat([oversampel_x, oversampel_y], axis=1)
plt.bar(counter.keys(), counter.values())
plt.show()


# Pre processing and Check data type

# In[10]:


data.dtypes


# In[11]:


feature_c1 = data[["has_online_delivery", "has_table_booking", "is_delivering_now","switch_to_order_menu"]]
feature_c2 = data[["cuisines","average_cost_for_two","price_range","votes"]]


# # clustering
# with k-means method

# In[12]:


k_means1 = KMeans(init = "random", n_clusters = 5, n_init = 20)
k_means1.fit(feature_c1)
labels1 = k_means1.labels_

k_means2 = KMeans(init = "random", n_clusters = 5, n_init = 20)
k_means2.fit(feature_c2)
labels2 = k_means2.labels_


# Sorting classfy label

# In[13]:


center1 = np.argsort(k_means1.cluster_centers_.sum(axis=1))
lut1 = np.zeros_like(center1)
lut1[center1] = np.arange(1,6)
data["classifier.1"]=lut1[labels1]

center2 = np.argsort(k_means2.cluster_centers_.sum(axis=1))
lut2 = np.zeros_like(center2)
lut2[center2] = np.arange(1,6)
data["classifier.2"]=lut2[labels2]


# ## Stick label from result clustering 

# In[14]:


data.loc[(data["classifier.1"] >= 3) & (data["classifier.2"] >= 3), 'output label'] = 'Delivery & Classy ambiance'
data.loc[(data["classifier.1"] >= 3) & (data["classifier.2"] < 3), 'output label'] = 'Delivery'
data.loc[(data["classifier.1"] < 3) & (data["classifier.2"] >= 3), 'output label'] = 'Classy ambiance'
data.loc[(data["classifier.1"] < 3) & (data["classifier.2"] < 3), 'output label'] = 'Not any'
data.head()


# In[17]:


zomato_csv = pd.read_csv('data/zomato.csv',encoding='ISO-8859-1')
zomato_csv = zomato_csv[["Restaurant ID","Restaurant Name","Country Code","City"]]
countryCode = pd.read_excel('data/Country-Code.xlsx')

zomato_csv.columns = ["id", "Restaurant Name", "Country Code", "City"]
countryCode.columns = ["Country Code", "Country"]
data["id"]=data["id"].astype(int)
data_merge = data[["id", "classifier.1", "classifier.2", "aggregate_rating", "output label"]] 


result = pd.merge(zomato_csv, countryCode, how="inner", on="Country Code")
result = result[["id", "Restaurant Name", "Country"]]
result = pd.merge(result, data_merge, how="inner", on="id")
result


# In[18]:


x = data[["classifier.1", "classifier.2"]]
y = data["aggregate_rating"]
x=np.asarray(x)
y=np.asarray(y).ravel()


# ## Train/Test
# Split data to two part train,test
# 
# The SVM algorithm offers a choice of kernel functions for performing its processing. Basically, mapping data into a higher dimensional space is called kernelling.
# 
# 1. Linear
# 2. Polynomial
# 3. Gaussian Radial basis function (RBF)
# 4. Sigmoid

# In[19]:


x1_train , x1_test , y1_train , y1_test = train_test_split(x , y , test_size=0.3 , random_state=5)


# Comparison kernel method of svm

# In[20]:


ks=[]
kernels = ["linear", "poly", "rbf"]
for i in kernels:
    clf_test=svm.SVC(kernel=i)
    clf_test.fit(x1_train,y1_train)
    yhat_test=clf_test.predict(x1_test)
    ks.append(cohen_kappa_score(y1_test, yhat_test))
ksd = {'linear': [ks[0]], 'poly': [ks[1]], 'gaussian rbf': [ks[2]]}
ksd


# Predict user rating with supervise

# In[21]:


clf=svm.SVC(kernel="rbf")
clf.fit(x1_train,y1_train)
yhat=clf.predict(x1_test)


# ## Confusion matrix and Analysis
# 
# In this section, we calculate the accuracy, but since the kappa score is a better benchmark for multi-tag data, we use it as follows:
# 
# - Kappa Less Than 0: "VERY BAD"
# - Kappa between 0 to 20: "BAD"
# - Kappa between 20 to 60: "MIDDLE"
# - Kappa between 20 to 60: "GOOD"
# - Kappa between 20 to 60: "EXCELLENT"

# In[22]:


from sklearn.metrics import classification_report, confusion_matrix
import itertools
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Greens):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


# In[23]:


# Compute confusion matrix
cnf_matrix = confusion_matrix(y1_test, yhat, labels=[1,2,3,4,5])
np.set_printoptions(precision=2)

print (classification_report(y1_test, yhat))
print("Kappa Score" , (cohen_kappa_score(y1_test, yhat)*100).astype(int))


plt.figure()
plot_confusion_matrix(cnf_matrix, classes=['class1','class2','class3','class4','class5'],normalize= False,  title='Confusion matrix')


# ## Accuracy curve Plot 

# In[24]:


def normalize_numpy(arr):
    norm = np.linalg.norm(arr)
    normal_array = arr/norm
    return normal_array
y_true = y1_test == yhat
y_pred = normalize_numpy(yhat)

RocCurveDisplay.from_predictions(y_true, y_pred)
plt.show()


# In[25]:


def predict_new_resturant(f1,f2,f3,f4,f5,f6,f7,f8):
    
    new_input1 = np.array([f1,f2,f3,f4])
    temp_dist = 0
    score1 = 0
    for i in range(5):
        dist = np.linalg.norm(new_input1 - k_means1.cluster_centers_[i])
        if(temp_dist > dist or i==0):
            temp_dist = dist
            score1 = i
    
    new_input2 = np.array([f5,f6,f7,f8])
    df_input2 = pd.DataFrame(data=new_input2.reshape(1, -1) , columns=["cuisines","average_cost_for_two","price_range","votes"])
    df_input2 = df_input2.values
    df_input2=scaler.transform(df_input2)
    temp_dist = 0
    score2 = 0
    for i in range(5):
        dist = np.linalg.norm(new_input2 - k_means2.cluster_centers_[i])
        if(temp_dist > dist or i==0):
            temp_dist = dist
            score2 = i 
            
    new_x = np.array([lut1[score1],lut2[score2]]).reshape(1, -1)
    new_yhat=clf.predict(new_x)
    output=[]
    output.append(lut1[score1])
    output.append(lut2[score2])
    output.append(new_yhat)
    return output


# In[26]:


f1=0                #has_online_delivery
f2=1                #has_table_booking
f3=0                #is_delivering_now
f4=0                #switch_to_order_menu
f5=3                #cuisines
f6=1200              #average_cost_for_two
f7=3                #price_range
f8=507/vote_max     #votes

output = predict_new_resturant(f1,f2,f3,f4,f5,f6,f7,f8)
cluster_input1= output[0]
cluster_input2= output[1]
new_yhat= output[2][0]

if(cluster_input1>=3 and cluster_input2>=3):
    print("This resturant have: 'Good Delivery & High Classy ambiance' And user rate predict %d out of 5" % (new_yhat))
elif(cluster_input1>=3 and cluster_input2<3):
    print("This resturant have: 'Good Delivery' And user rate predict %d out of 5" % (new_yhat))
elif(cluster_input1<3 and cluster_input2>=3):
    print("This resturant have: 'High Classy ambiance' And user rate predict %d out of 5" % (new_yhat))
elif(cluster_input1<3 and cluster_input2<3):
    print("This resturant have: 'No Delivery No classy ambiance' And user rate predict %d out of 5" % (new_yhat))


# In[ ]:




