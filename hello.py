import sklearn  # pre existing dataset use korte parbo sklearn er

import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import numpy as np
from sklearn import datasets, linear_model

diabetes = datasets.load_diabetes()
#print(diabetes.keys())# diabetes er under e ki ki dataset ache ta dekhai
#'data', 'target', 'frame', 'DESCR', 'feature_names', 'data_filename',
# 'target_filename', 'data_module'
#print (diabetes.data) # numpy arrays
#print (diabetes.DESCR) # DESCR ki dekhai
diabetes_X = diabetes.data[:, np.newaxis, 2]  # linear regression er jonno sudhu matro ditiyo column neya hocche
# 2  no index er feature k column banaya dice array akare
diabetes_X_Train = diabetes_X[:-30]  # sesh 30
diabetes_X_Test = diabetes_X[-30:]  # surur 20
diabetes_Y_Train = diabetes.target[:-30]
diabetes_Y_Test = diabetes.target[-30:]
model = linear_model.LinearRegression()  # jlinear model jeta upure likhci
model.fit(diabetes_X_Train, diabetes_Y_Train)
diabetes_Y_predicted = model.predict(diabetes_X_Test)
print(' Mean squarred  error is: ', mean_squared_error(diabetes_Y_Test, diabetes_Y_predicted))
print("Weights:", model.coef_)
print("Intercept:", model.intercept_)
plt.scatter(diabetes_X_Test, diabetes_Y_Test, color='red')
plt.plot(diabetes_X_Test, diabetes_Y_predicted, color='blue')
plt.show()
