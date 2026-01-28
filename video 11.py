import numpy as np
import matplotlib.pyplot as plt
import sklearn
from sklearn.metrics import mean_squared_error
import numpy as np
from sklearn import datasets, linear_model

diabetes_X = np.array([[1],[2],[3]]) # multiple regression er jonno sudhu matro ditiyo column neya hocche
# 2  no index er feature k column banaya dice array akare
diabetes_X_Train = diabetes_X # sesh 30
diabetes_X_Test = diabetes_X # surur 20
diabetes_Y_Train = np.array([[3],[2],[4]])
diabetes_Y_Test = np.array([[3],[2],[4]])
model = linear_model.LinearRegression()  # jlinear model jeta upure likhci
model.fit(diabetes_X_Train, diabetes_Y_Train)
diabetes_Y_predicted = model.predict(diabetes_X_Test)
print(' Mean squarred  error is: ', mean_squared_error(diabetes_Y_Test, diabetes_Y_predicted))
print("Weights:", model.coef_)
print("Intercept:", model.intercept_)
plt.scatter(diabetes_X_Test, diabetes_Y_Test, color='red')
plt.plot(diabetes_X_Test, diabetes_Y_predicted, color='blue')
plt.show()