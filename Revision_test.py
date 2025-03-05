import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

df =pd.read_csv("logistic_regression.csv")
print(df)

# """below code is used to return only last 5 rows of the csv file """
# print(df.tail())

# """below code is used to get the specific data from the rows"""
# print(df[4:10])

# """10th rows to all the rows"""
# print(df[10:])

# """sliciling the row using locator function of pandas"""
# print(df.loc[4:10])

# """slicilng the row using locater function of pandas for specific columns too """
# print(df.loc[4:10,["Studied","Slept"]])

# """slicing the row and columns based on its index position"""
# print(df.iloc[4:10,0:2])

# """how to pick data's of specific column"""
# df= df['Slept']
# print(df)

# """to find data types of entire dataset"""
# columns_type = df.dtypes
# print(columns_type)

# """detailing about the entire dataset"""
# column_data = df.describe()
# print(column_data)

# X = [11,12,13,14,15]
# Y = [2,4,6,8,10]

# plt.plot()
# plt.plot(X,Y)
# plt.scatter(X,Y,marker="X",color="yellow",label="my simple plot",alpha=0.7)
# plt.show()

df_info = df.info()
print(df_info)

"""to check the outlier in the give input"""
fig, axes = plt.subplots(3,1, figsize=(5,5))

axes[0].boxplot(df['Studied'])
axes[1].boxplot(df['Slept'])
axes[2].boxplot(df['Passed'])

plt.tight_layout()
plt.show()




