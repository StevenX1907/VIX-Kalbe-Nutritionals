#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# ## Read Files

# In[2]:


customer = pd.read_csv('Customer.csv', delimiter = ';')
product = pd.read_csv('Product.csv', delimiter = ';')
store = pd.read_csv('Store.csv', delimiter = ';')
transaction = pd.read_csv('Transaction.csv', delimiter = ';')


# In[3]:


customer.shape, product.shape, store.shape, transaction.shape


# ## Data Cleaning

# In[5]:


customer.isnull().sum()


# In[10]:


customer = customer.dropna()
customer


# In[12]:


customer['Income'] = customer['Income'].replace(',','.', regex = True).astype('float')

customer


# In[13]:


product.isnull().sum()


# In[14]:


product


# In[15]:


store.isnull().sum()


# In[16]:


store.head()


# In[17]:


store['Latitude'] = store['Latitude'].replace(',','.', regex = True).astype('float')
store['Longitude'] = store['Longitude'].replace(',','.', regex = True).astype('float')
store


# In[18]:


transaction.isnull().sum()


# In[19]:


transaction.head()


# In[22]:


transaction['Date'] = pd.to_datetime(transaction['Date'])
transaction


# In[21]:


df_merge = pd.merge(customer, transaction, on = ['CustomerID'])
df_merge = pd.merge(df_merge, store, on = ['StoreID'])
df_merge = pd.merge(df_merge, product.drop(columns = ['Price']), on = ['ProductID'])
df_merge = df_merge.sort_values(by='Date').reset_index(drop = True)
df_merge.head()


# # ARIMA

# In[24]:


df_regression = df_merge.groupby(['Date']).agg({
    'Qty':'sum'
}).reset_index()
df_regression


# In[25]:


plt.figure(figsize = (20,6))
plt.plot(df_regression['Date'], df_regression['Qty'], linestyle = '-')
plt.title('Time Series Plot')
plt.xlabel('Date')
plt.ylabel('Value')
plt.show()


# In[26]:


import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA


# ### Data Trend and Seasonality

# In[27]:


reg_decompose = seasonal_decompose(df_regression.set_index('Date'))

plt.figure(figsize = (10,8))

plt.subplot(411)
reg_decompose.observed.plot(ax = plt.gca())
plt.title('Observed')
plt.subplot(412)
reg_decompose.trend.plot(ax = plt.gca())
plt.title('Trend')
plt.subplot(413)
reg_decompose.seasonal.plot(ax = plt.gca())
plt.title('Seasonality')
plt.subplot(414)
reg_decompose.resid.plot(ax = plt.gca())
plt.title('Residual')
plt.tight_layout()


# ### Data Stationarity Check
# 
# H0 = Data is not stationary <br>
# H1 = Data is stationary <br>
# α = 0.05

# In[28]:


result = adfuller(df_regression['Qty'])
print('ADF Statistic: %f' % result[0])
print('p-value: %.2f' % result[1])
print('Num of Lags: %f' % result[1])
print('Critical Values:')
for key, value in result[4].items():
  print('\t%s: %.3f' % (key, value))

if (result[1]) <= 0.05:
    print('\nReject H0. Data is stationary')
else:
    print('\nAccept H0. Data is not stationary')


# d = 0

# In[29]:


from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

original_pacf = plot_pacf(df_regression['Qty'])
original_acf = plot_acf(df_regression['Qty'])


# ### Splitting Data
# 
# 80% for training and 20% for testing

# In[30]:


split_size = round(df_regression.shape[0] * 0.8)
data_train = df_regression[:split_size]
data_test = df_regression[split_size:].reset_index(drop = True)
data_train.shape, data_test.shape


# In[31]:


data_train


# In[32]:


data_test


# In[33]:


plt.figure(figsize =(20,5))
sns.lineplot(data = data_train, x = data_train['Date'], y = data_train['Qty'])
sns.lineplot(data = data_test, x = data_test['Date'], y = data_train['Qty'])


# In[34]:


train_pacf = plot_pacf(data_train['Qty'], lags = 30)
train_acf = plot_acf(data_train['Qty'], lags = 30)


# In[35]:


pip install pmdarima


# In[36]:


from pmdarima import auto_arima
# Ignore harmless warnings
import warnings
warnings.filterwarnings ("ignore")

stepwise_fit = auto_arima(data_train['Qty'],trace = True, 
                          suppress_warnings = True)

print(stepwise_fit.summary())


# In[37]:


from pandas.plotting import autocorrelation_plot

autocorrelation_plot(data_train['Qty']).set_xlim([0, 100])


# In[38]:


from statsmodels.tsa.arima.model import ARIMA

y = data_train['Qty']

model_1 = ARIMA(y, order = (0,0,0))
model_2 = ARIMA(y, order = (28,0,28))
model_3 = ARIMA(y, order = (44,0,44))


# In[39]:


model_1 = model_1.fit()
print(model_1.summary())


# In[40]:


model_2 = model_2.fit()
print(model_2.summary())


# In[41]:


model_3 = model_3.fit()
print(model_3.summary())


# In[44]:


data_train = data_train.set_index('Date')
data_test = data_test.set_index('Date')

y_pred_1 = model_1.get_forecast(len(data_test))
y_pred_2 = model_2.get_forecast(len(data_test))
y_pred_3 = model_3.get_forecast(len(data_test))

y_pred_df_1 = y_pred_1.conf_int()
y_pred_df_1['Predictions'] = model_1.predict(start = y_pred_df_1.index[0], end = y_pred_df_1.index[-1])
y_pred_df_1.index = data_test.index
y_pred_out_1 = y_pred_df_1['Predictions']

y_pred_df_2 = y_pred_2.conf_int()
y_pred_df_2['Predictions'] = model_2.predict(start = y_pred_df_2.index[0], end = y_pred_df_2.index[-1])
y_pred_df_2.index = data_test.index
y_pred_out_2 = y_pred_df_2['Predictions']

y_pred_df_3 = y_pred_3.conf_int()
y_pred_df_3['Predictions'] = model_3.predict(start = y_pred_df_3.index[0], end = y_pred_df_3.index[-1])
y_pred_df_3.index = data_test.index
y_pred_out_3 = y_pred_df_3['Predictions']

plt.figure(figsize = (30,7))
plt.plot(data_train['Qty'])
plt.plot(data_test['Qty'], color = 'orange', label = 'Actual')
plt.plot(y_pred_out_1, color = 'red', label = 'Model 1 Predictions')
plt.plot(y_pred_out_2, color = 'green', label = 'Model 2 Predictions')
plt.plot(y_pred_out_3, color = 'purple', label = 'Model 3 Predictions')
plt.legend()


# In[45]:


y_actual = data_test['Qty']

plt.figure(figsize=(20, 6))
plt.plot(data_test.index, y_actual, label = 'Actual', color = 'orange')
plt.plot(data_test.index, y_pred_out_1, label= 'Model 1 Predictions', linestyle='--', color = 'red')
plt.plot(data_test.index, y_pred_out_2, label= 'Model 2 Predictions', linestyle='--', color = 'green')
plt.plot(data_test.index, y_pred_out_3, label= 'Model 3 Predictions', linestyle='--', color = 'purple')
plt.xlabel('Date')
plt.ylabel('Qty')
plt.legend()
plt.title('Actual vs. Predicted')
plt.show()


# In[46]:


forecast_periods = 150

y_pred_2_future = model_2.get_forecast(steps=forecast_periods)
y_pred_df_2_future = y_pred_2_future.conf_int()
y_pred_df_2_future['Predictions'] = y_pred_2_future.predicted_mean
y_pred_df_2_future.index = pd.date_range(start=data_train.index[-1], periods=forecast_periods+1, closed='right')
y_pred_out_2_future = y_pred_df_2_future['Predictions']

y_pred_3_future = model_3.get_forecast(steps=forecast_periods)
y_pred_df_3_future = y_pred_3_future.conf_int()
y_pred_df_3_future['Predictions'] = y_pred_3_future.predicted_mean
y_pred_df_3_future.index = pd.date_range(start=data_train.index[-1], periods=forecast_periods+1, closed='right')
y_pred_out_3_future = y_pred_df_3_future['Predictions']

plt.figure(figsize=(30, 7))
plt.plot(data_test['Qty'], color='orange', label='Actual')
plt.plot(y_pred_out_2, color='green', label='Model 2 Predictions')
plt.plot(y_pred_out_3, color='purple', label='Model 3 Predictions')
plt.plot(y_pred_out_2_future, color='lime', linestyle='-', label='Model 2 Future Forecast')
plt.plot(y_pred_out_3_future, color='magenta', linestyle='-', label='Model 3 Future Forecast')
plt.legend()


# In[47]:


from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error

mae_1 = mean_absolute_error (data_test, y_pred_out_1)
mse_1 = mean_squared_error (data_test, y_pred_out_1)
rmse_1 = np.sqrt (mean_squared_error(data_test, y_pred_out_1))
mape_1 = mean_absolute_percentage_error(data_test, y_pred_out_1)*100

mae_2 = mean_absolute_error (data_test, y_pred_out_2)
mse_2 = mean_squared_error (data_test, y_pred_out_2)
rmse_2 = np.sqrt (mean_squared_error(data_test, y_pred_out_2))
mape_2 = mean_absolute_percentage_error(data_test, y_pred_out_2)*100

mae_3 = mean_absolute_error (data_test, y_pred_out_3)
mse_3 = mean_squared_error (data_test, y_pred_out_3)
rmse_3 = np.sqrt (mean_squared_error(data_test, y_pred_out_3))
mape_3 = mean_absolute_percentage_error(data_test, y_pred_out_3)*100

print("Model 1")
print(f"Mean Absolute Error (MAE): {mae_1:.2f}")
print(f"Mean Squared Error (MSE): {mse_1:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse_1:.2f}")
print(f"Mean Absolute Percentage Error (MAPE): {mape_1:.2f}%")

print("\nModel 2")
print(f"Mean Absolute Error (MAE): {mae_2:.2f}")
print(f"Mean Squared Error (MSE): {mse_2:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse_2:.2f}")
print(f"Mean Absolute Percentage Error (MAPE): {mape_2:.2f}%")

print("\nModel 3")
print(f"Mean Absolute Error (MAE): {mae_3:.2f}")
print(f"Mean Squared Error (MSE): {mse_3:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse_3:.2f}")
print(f"Mean Absolute Percentage Error (MAPE): {mape_3:.2f}%")


# # K-Means Clustering

# In[49]:


from sklearn import preprocessing
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


# In[50]:


df_clustering = df_merge.groupby(['CustomerID']).agg({
    'TransactionID':'count',
    'Qty':'sum',
    'TotalAmount':'sum'
}).reset_index()
df_clustering


# In[51]:


import seaborn as sns
import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(10,8))
sns.scatterplot(data=df_clustering, x='Qty', y='TotalAmount')


# In[52]:


pip install threadpoolctl==3.1.0


# ## Data Scaling

# In[53]:


from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
temp = df_clustering.pop('CustomerID')

scaler = StandardScaler()
scaler.fit(df_clustering)
scaled_data = scaler.transform(df_clustering)
scaled_data


# ## Method 1: Elbow Method

# In[54]:


distortions = []
K = range(1,11)
for k in K:
    kmeanModel = KMeans(n_clusters=k)
    kmeanModel.fit(scaled_data)
    distortions.append(kmeanModel.inertia_)


# In[55]:


plt.figure(figsize=(16,8))
plt.plot(K, distortions, 'o-')
plt.xlabel('K-Values')
plt.ylabel('Inertia')
plt.title('The Elbow Method showing the optimal k')
plt.show()


# k = 3

# ## Method 2: Silhouette Score

# In[56]:


pip install yellowbrick


# In[57]:


from yellowbrick.cluster import SilhouetteVisualizer

fig, ax = plt.subplots(2, 2, figsize=(15,8))
KS = [2, 3, 4, 5]
for k in KS:
    km = KMeans(n_clusters=k, 
                init='k-means++', 
                n_init=10, 
                max_iter=100, 
                random_state=42)
    q, mod = divmod(k, 2)
    visualizer = SilhouetteVisualizer(km, 
                                      colors='yellowbrick', 
                                      ax=ax[q-1][mod])
    visualizer.fit(scaled_data) 


# k = 3

# In[59]:


kmeans_model = KMeans(n_clusters = 3)
kmeans_model.fit(scaled_data)


# In[61]:


df_clustering['Cluster'] = kmeans_model.labels_
df_clustering['CustomerID'] = temp
df_clustering


# In[62]:


plt.figure(figsize = (10, 7))
plt.scatter(df_clustering['Qty'],
            df_clustering['TotalAmount'],
            c = df_clustering['Cluster'],
           cmap = 'viridis')
plt.xlabel('Qty')
plt.ylabel('Total Amount')
plt.title('Customer Segmentation using KMeans')
plt.show()


# In[63]:


pip install plotly


# In[64]:


import plotly.express as px

fig = px.scatter_3d(df_clustering, x='Qty', y='TotalAmount', z='Cluster',
                     color='Cluster', symbol='Cluster', opacity=0.7)

fig.update_layout(scene=dict(xaxis_title='Qty',
                             yaxis_title='Total Amount',
                             zaxis_title='Cluster'),
                  title='Customer Segmentation using KMeans',
                  margin=dict(l=0, r=0, b=0, t=40))

fig.show()


# In[65]:


import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure(figsize=(15, 7))
ax = fig.add_subplot(111, projection='3d')

scatter = ax.scatter(df_clustering['Qty'],
                      df_clustering['TotalAmount'],
                      df_clustering['Cluster'],
                      c=df_clustering['Cluster'], cmap='viridis')

ax.set_xlabel('Qty')
ax.set_ylabel('Total Amount')
ax.set_zlabel('Cluster')
ax.set_title('Customer Segmentation using KMeans')

colorbar = fig.colorbar(scatter, ax=ax)
colorbar.set_label('Cluster')

plt.show()


# In[66]:


cluster_stats = df_clustering.groupby(['Cluster']).agg({
    'CustomerID':'count',
    'TransactionID': ['mean', 'median'],  
    'Qty': ['mean', 'median'],            
    'TotalAmount': ['mean', 'median']     
}) 
cluster_stats


# In[68]:


import matplotlib.cm as cm

n_clusters = cluster_stats['CustomerID']['count']
colors = cm.get_cmap('viridis', len(n_clusters))
pie_colors = colors(range(len(n_clusters)))
text_colors = ['white', 'black', 'white']

plt.figure(figsize=(8, 8))
plt.pie(n_clusters, labels=n_clusters.index, autopct='%1.1f%%', 
        startangle=140, colors = pie_colors, textprops={'color': 'white'})
plt.title('Customer Distribution by Cluster')
plt.axis('equal')
plt.legend(n_clusters.index, title="Cluster", loc="center left", bbox_to_anchor=(1, 0, 0.5, 1))
plt.show()

