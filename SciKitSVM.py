from pmlb import fetch_data, classification_dataset_names
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score

import matplotlib.pyplot as plt
import seaborn as sb
##for ease of use
class supportVector:
    
    def __init__(self, kernel_type):
        self.svm1 = svm.SVC(kernel = kernel_type, cache_size = 1000)
        self.test_scores = []
        self.matrices = []
        self.predictions = []
    
    def predictf(self, x, y):
        self.predictions = self.svm1.predict(x)
        self.test_scores.append(accuracy_score(self.predictions, y))
        self.matrices.append(confusion_matrix(y, self.predictions))
        
        
        
##initiating all necessary variables 
rbf_svm = supportVector('rbf')
polynomial_svm = supportVector('poly')
linear_svm = supportVector('linear')
sig_svm = supportVector('sigmoid')

supportVectorArray = [rbf_svm, polynomial_svm, linear_svm, sig_svm]

##preparing the data
X,y = fetch_data('banana', return_X_y=True)

train_X, test_X, train_y, test_y = train_test_split(X,y)

##training the models
for svmachine in supportVectorArray:
    svmachine.svm1.fit(train_X, train_y)    
    
    ##predict, test, and create confusion matrix
    svmachine.predictf(test_X, test_y)
    

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





sb.boxplot(data=[supportVectorArray[0].test_scores, supportVectorArray[1].test_scores, supportVectorArray[2].test_scores, supportVectorArray[3].test_scores], notch=True)
plt.xticks([0, 1,2,3], ['RBFSVM', 'PolynomialSVM', 'LinearSVM', 'SigmoidSVM'])
plt.ylabel('Test Accuracy')
plt.show()