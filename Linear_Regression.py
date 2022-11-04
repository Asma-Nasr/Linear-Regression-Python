import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

df = pd.read_csv("homeprices.csv")

#Visualize the data
plt.xlabel('Area')
plt.ylabel('Price')
plt.scatter(df.Area, df.Price, color='red',marker='+' )
# Fitting the model to the data
reg = LinearRegression()
reg.fit(df[['Area']],df.Price)
# Predict new data
reg.predict([[3300]])

# The y intercept
print(reg.intercept_)

# The regression coefficient
print(reg.coef_)
# mx + b
print(135.788*3300+180616.438)