#!/usr/bin/env python
# coding: utf-8

# In[65]:


import numpy as np
import pandas as pd
import matplotlib as mpl
import seaborn as sns
import scipy as sp
import matplotlib.pyplot as plt


# In[43]:


file_path = r"C:\Users\Cyber World\Desktop\orderdataset.csv"
s_data = pd.read_csv(file_path, delimiter=";")

# Display settings 
pd.set_option('display.max_columns', None)  
pd.set_option('display.max_rows', 80) 
pd.set_option('display.width', 1000) 
pd.set_option('display.float_format', '{:.2f}'.format)  
styled_df = s_data.head(80).style.set_table_styles(
    [{'selector': 'th', 'props': [('text-align', 'left')]}]
)
styled_df



# In[49]:


#Missing values 
missing_values = s_data.isnull().sum()
print(missing_values)


# In[53]:


# Fill missing values
s_data['product_weight_gram'].fillna(s_data['product_weight_gram'].median(), inplace=True)
print(s_data.isnull().sum())


# In[55]:


# Check for duplicate 
duplicates = s_data.duplicated().sum()
print(f"Number of duplicate rows: {duplicates}")
s_data.drop_duplicates(inplace=True)


# In[56]:


# Convert 'purchase_date' to datetime
s_data['purchase_date'] = pd.to_datetime(s_data['purchase_date'], format='%d/%m/%Y')
print(s_data.dtypes)


# In[57]:


# Check for non-numeric values
for column in ['quantity', 'price', 'freight_value', 'product_weight_gram']:
    non_numeric_values = s_data[pd.to_numeric(s_data[column], errors='coerce').isna()]
    if not non_numeric_values.empty:
        print(f"Non-numeric values found in {column}:")
        print(non_numeric_values)
print(s_data['order_status'].unique())
print(s_data['payment_type'].unique())


# In[58]:


# Save the cleaned data to a new CSV file
cleaned_file_path = r"C:\Users\Cyber World\Desktop\cleaned_orderdataset.csv"
s_data.to_csv(cleaned_file_path, index=False)


# In[59]:


# Descriptive statistics
# Total sales amount
total_sales_amount = s_data['price'].sum()
print(f"Total Sales Amount: {total_sales_amount}")


# In[60]:


# Average sales amount
average_sales_amount = s_data['price'].mean()
print(f"Average Sales Amount: {average_sales_amount}")


# In[61]:


# Total number of orders
total_orders = s_data['order_id'].nunique()
print(f"Total Number of Orders: {total_orders}")


# In[62]:


# Average order value
average_order_value = s_data.groupby('order_id')['price'].sum().mean()
print(f"Average Order Value: {average_order_value}")


# In[63]:


# Freight value stats
freight_value_stats = s_data['freight_value'].agg(['sum', 'mean', 'min', 'max'])
print("Freight Value Statistics:")
print(freight_value_stats)


# In[67]:


# C0rrelation Analysis
correlation_matrix = s_data[['quantity', 'price', 'freight_value', 'product_weight_gram']].corr()
print("Correlation Matrix:")
print(correlation_matrix)


# In[64]:


# Comparative Analysis
sales_by_category = s_data.groupby('product_category_name')['price'].sum().sort_values(ascending=False)
sales_by_payment_type = s_data.groupby('payment_type')['price'].sum().sort_values(ascending=False)
print("Sales by Product Category:")
print(sales_by_category)
print("\nSales by Payment Type:")
print(sales_by_payment_type)


# In[66]:


# Distribution of prices
plt.figure(figsize=(10, 6))
plt.hist(s_data['price'], bins=30, edgecolor='black')
plt.title('Distribution of Prices')
plt.xlabel('Price')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()
plt.figure(figsize=(10, 6))
s_data['product_weight_gram'].plot(kind='kde')
plt.title('Kernel Density Estimation of Product Weights')
plt.xlabel('Product Weight (grams)')
plt.grid(True)
plt.show()


# In[69]:


# Trend Analysis
s_data['purchase_month'] = s_data['purchase_date'].dt.month
s_data['purchase_year'] = s_data['purchase_date'].dt.year
monthly_sales = s_data.groupby('purchase_month')['price'].sum()
yearly_sales = s_data.groupby('purchase_year')['price'].sum()
print("Monthly Sales Trend:")
print(monthly_sales)
print("\nYearly Sales Trend:")
print(yearly_sales)


# In[70]:


# Boxplot of sales by payment type
plt.figure(figsize=(10, 6))
sns.boxplot(x='payment_type', y='price', data=s_data)
plt.title('Sales Distribution by Payment Type')
plt.xlabel('Payment Type')
plt.ylabel('Price')
plt.grid(True)
plt.show()


# In[71]:


# Yearly sales Trends
yearly_sales = s_data.groupby('purchase_year')['price'].sum()
plt.figure(figsize=(10, 6))
yearly_sales.plot(kind='line', marker='o', color='b', linestyle='-', linewidth=2)
plt.title('Yearly Sales Trend')
plt.xlabel('Year')
plt.ylabel('Total Sales Amount')
plt.grid(True)
plt.xticks(yearly_sales.index)
plt.show()


# In[74]:


plt.savefig('price_distribution.png')  


# In[ ]:


plt.savefig('sales_by_payment_type.png')


# In[ ]:


plt.savefig('yearly_sales_trend.png') 

