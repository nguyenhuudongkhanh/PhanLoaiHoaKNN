# Bước 1: Khai Báo Các Thư Viện Cần Thiết
import numpy as np
import matplotlib.pyplot as plt
from sklearn import neighbors, datasets

iris = datasets.load_iris()
iris_X = iris.data
iris_y = iris.target
print ('Số lớp: %d' %len(np.unique(iris_y)))
print ('Số lượng điểm dữ liệu: %d' %len(iris_y))

# Bước 2: Load dữ liệu và hiển thị vài dữ liệu mẫu
X0 = iris_X[iris_y == 0,:]
print ('\nMẫu từ lớp 0:\n', X0[:50,:])

X1 = iris_X[iris_y == 1,:]
print ('\nMẫu từ lớp 1:\n', X1[:50,:])

X2 = iris_X[iris_y == 2,:]
print ('\nMẫu từ lớp 2:\n', X2[:50,:])
# Bước 3: Tách traning set và test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
     iris_X, iris_y, test_size=45)

print ("Training size: %d" %len(y_train))
print ("Test size    : %d" %len(y_test))

# # Bước 4: KNN dự đoán
clf = neighbors.KNeighborsClassifier(n_neighbors = 1, p = 2)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

print ("In kết quả cho 30 điểm dữ liệu thử nghiệm:")
print ("Các nhãn được dự đoán: ", y_pred[30:60])
print ("Thực địa   : ", y_test[30:60])

from sklearn.metrics import accuracy_score
print ("Accuracy of 1NN: %.2f %%" %(100*accuracy_score(y_test, y_pred)))

clf = neighbors.KNeighborsClassifier(n_neighbors = 9, p = 2)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

print ("In kết quả cho 30 điểm dữ liệu thử nghiệm:")
print ("Các nhãn được dự đoán: ", y_pred[30:60])
print ("Thực địa   : ", y_test[30:60])

from sklearn.metrics import accuracy_score
print ("Accuracy of 1NN: %.2f %%" %(100*accuracy_score(y_test, y_pred)))


clf = neighbors.KNeighborsClassifier(n_neighbors = 9, p = 2, weights = 'distance')
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

print ("Accuracy of 10NN (1/distance weights): %.2f %%" %(100*accuracy_score(y_test, y_pred)))


