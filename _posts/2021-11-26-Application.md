---
layout: pos
title:  "Least Squares Application"
image:  /images/App-img1.png
image1:  /images/App-img2.png
image2:  /images/App-img3.png
image3:  /images/App-img4.png

toc: true
---

For this task we will be analysing the change in Lahore's Air Quality Index (AQI) over the year 2021 and try to predict trends based on the data that we have. Here $y$ will represent the AQI of the day and $x$ will represent the days, with  $x$ = 1 corresponding to 01/01/2021 and $x$ = 274 corresponding to 30/09/2021.


Least squares approximation is a method to estimate the true value of parameters given a measurements that have a lot of noise in them. The least squares solution will be a "best approximate solution" that minimizes sum of squared distances. For an $Ax = y$, with A being the model and x being the input given, we wish to find a model that minimizes the $\mathbb{L_2}$ square norm error.

  $$ \Vert{\mathbf{A}\hat{\mathbf{x}} - \mathbf{y}}\Vert_2^2$$

Our goal will be to apply least squares approximation for different models on the noisy data that we hare working with.

**1. Importing Libaries**

We will be using four Library's in python `numpy`, `matplotlib`, `pandas` and `Sklearn`.

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as
```

**2. Loading and Viewing Data**

```python
df = pd.read_csv("lahore-aqi-data.csv")
df = df[(df['date'] > '2021-01-01')]
df.plot(x = 'date', y = ' pm25')
plt.show()
```

<div style="text-align: center;"> <img src="{{page.image | relative_url}}" height="220" width="400"></div>

**3. Splitting the data and reshaping data**

We will splitting our data into training and test set. Where 80% of the data will go to training and the rest of the 20% will go to test set. We will also be using `np.linspace` to transform our x values which are in the form of dates and unsuitable as input to days.

```python
X = np.linspace(1,274,273)
y = df[' pm25'].values
X = X.reshape(-1, 1)
y = y.reshape(-1, 1)
```

```python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
```

**4. Least Squares with 3 different models**


```python
from sklearn.preprocessing import PolynomialFeatures

def polymonial_regression(deg):
    poly_reg = PolynomialFeatures(degree=deg)
    X_r = poly_reg.fit_transform(X)
    pl_reg = LinearRegression()
    pl_reg.fit(X_r, y)
    plt.scatter(X, y, color='green')
    plt.plot(X, pl_reg.predict(poly_reg.fit_transform(X)), color='red')
    plt.title("Fitting polynomial of Degree {}".format(deg))
    plt.xlabel('Days')
    plt.ylabel('AQI')

    return
```
<div style="float: right;"> <img src="{{page.image2 | relative_url}}" height="220" width="350"></div>
<div style="float: left;"> <img src="{{page.image1
 | relative_url}}" height="220" width="380"></div>

 <div style="text-align: center;"> <img src="{{page.image3
  | relative_url}}" height="220" width="500"></div>


**5. Prediction**

```python
def predict(deg, day):
    poly_reg = PolynomialFeatures(degree=deg)
    X_r = poly_reg.fit_transform(X)
    pl_reg = LinearRegression()
    pl_reg.fit(X_r, y)
    return pl_reg.predict(poly_reg.fit_transform([[day]]))
```

```python
predict(1, 300)  
#output is array([[77.26266677]])
predict(3, 300)  
#output is array([[198.63029506]])
predict(6, 300)    
#output is array([[410.64869444]])
```

We can see that higher degree polynomials cause _overfitting_.This happens when the model we have built is more complex than the actual model. When we try and predict the AQI for the date 27th October 2021 which is the 300th day of the year. The actual value of the AQI was 192. From our linear model we obtained the result of 77 and from our poylnomial of degree 6 we obtained 410 AQI while from our model with degree 3 we got 198 which is closest to our actual value.
