from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier
#Loading datasets
iris=datasets.load_iris()
#print description and features
features=iris.data
labels=iris.target
print(iris.DESCR)
print(features[0],labels[0])
#training Classifier
clf=KNeighborsClassifier()
clf.fit(features,labels)
preds=clf.predict([[1,1,1,1]])
print(preds)
