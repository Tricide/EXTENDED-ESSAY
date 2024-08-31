from pmlb import fetch_data, classification_dataset_names
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score

import matplotlib.pyplot as plt
import seaborn as sb


rbf_svm_test_scores = []
polynomial_svm_test_scores = []
linear_svm_test_scores = []
sig_svm_test_scores = []

rbf_svm = svm.SVC(kernel = 'rbf', cache_size = 1000)
polynomial_svm = svm.SVC(kernel = 'poly', cache_size = 1000)
linear_svm = svm.SVC(kernel = 'linear', cache_size = 1000)
sig_svm = svm.SVC(kernel = 'sigmoid', cache_size = 1000)

X,y = fetch_data('banana', return_X_y=True)

train_X, test_X, train_y, test_y = train_test_split(X,y)

rbf_svm.fit(train_X, train_y)
polynomial_svm.fit(train_X, train_y)
linear_svm.fit(train_X, train_y)
sig_svm.fit(train_X, train_y)

rbf_predicted=rbf_svm.predict(test_X)
polynomial_predicted=polynomial_svm.predict(test_X)
linear_predicted=linear_svm.predict(test_X)
sig_predicted=sig_svm.predict(test_X)

rbf_svm_test_scores.append(accuracy_score(rbf_predicted))
polynomial_svm_test_scores.append(accuracy_score(polynomial_predicted))
linear_svm_test_scores.append(accuracy_score(linear_predicted))
sig_svm_test_scores.append(accuracy_score(sig_predicted))

'''
for classification_dataset in classification_dataset_names:
    
    rbf_svm = svm.SVC(kernel = 'rbf')
    polynomial_svm = svm.SVC(kernel = 'poly')

    X,y = fetch_data(classification_dataset, return_X_y=True)

    train_X, test_X, train_y, test_y = train_test_split(X,y)

    rbf_svm.fit(train_X, train_y)
    polynomial_svm.fit(train_X, train_y)


    rbf_svm_test_scores.append(rbf_svm.score(test_X, test_y))
    polynomial_svm_test_scores.append(polynomial_svm.score(test_X, test_y))
'''


print(confusion_matrix(train_y, rbf_predicted))
print(confusion_matrix(train_y, polynomial_predicted))
print(confusion_matrix(train_y, linear_predicted))
print(confusion_matrix(train_y, sig_predicted))



sb.boxplot(data=[rbf_svm_test_scores, polynomial_svm_test_scores, linear_svm_test_scores, sig_svm_test_scores], notch=True)
plt.xticks([0, 1,2,3], ['RBFSVM', 'PolynomialSVM', 'LinearSVM', 'SigmoidSVM'])
plt.ylabel('Test Accuracy')
plt.show()