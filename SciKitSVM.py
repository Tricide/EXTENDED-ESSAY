from pmlb import fetch_data, classification_dataset_names
from sklearn import svm
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, f1_score, precision_score, balanced_accuracy_score, roc_auc_score, roc_curve, auc, precision_recall_curve


import matplotlib.pyplot as plt
import seaborn as sb
import openpyxl

import time
##for ease of use
class supportVector:
    
    def __init__(self, kernel_type, c, gamma, degree):
        self.svm1 = svm.SVC(kernel = kernel_type, cache_size = 1000, probability = True, )
        self.matrix = []
        self.predictions = []
        self.PCAPredictions = []
        self.kernelPCAPredictions = []
        self.probabilities = [] 
        self.tests = {"f1score" : 0,
                      "accuracy": 0,
                      "balancedAccuracy": 0,
                      "specificity" : 0,
                      "precision": 0,
                      "recall": 0,
                      "rocauc": 0}
    
    def predictf(self, x, y):
        self.predictions = self.svm1.predict(x)
        self.probabilities = self.svm1.predict_proba(x)
        self.matrix = confusion_matrix(y, self.predictions).ravel()
        self.tests["f1score"] = f1_score(y, self.predictions)
        self.tests["accuracy"] = accuracy_score(y, self.predictions)
        self.tests["balancedAccuracy"] = balanced_accuracy_score(y, self.predictions)
        self.tests["precision"] = self.matrix[0]/(self.matrix[0]+self.matrix[1])
        self.tests["precision"] = precision_score(y,self.predictions)
        self.tests["recall"] = recall_score(y, self.predictions)
        self.tests["rocauc"] = roc_auc_score(y, self.predictions)
        
    
        
        
datasets = ["banana", "clean2", "parity5", "adult", "GAMETES_Epistasis_2_Way_1000atts_0.4H_EDM_1_EDM_1_1"]        
gammaValues = [0.001, 0.01, 0.1, 1, 10, 100]
cValues = [0.001, 0.01, 0.1, 1, 10, 100]
degreeValues = [2,3,4,5,6,7,8,9,10,11]

supportVectorArray = []
for c in cValues:
    linear_svm = supportVector('linear', c, 0, 0)
    supportVectorArray.append(linear_svm)
    for gamma in gammaValues:
        rbf_svm = supportVector('rbf', c, gamma, 1)
        sig_svm = supportVector('sigmoid', c, gamma, 1)
        supportVectorArray.append(rbf_svm)
        supportVectorArray.append(sig_svm)
        for degree in degreeValues:
            polynomial_svm = supportVector('polynomial', c, gamma, degree)
            supportVectorArray.append(polynomial_svm)
            
         
      
def run(supportVectorMachine):            
    for dataset in datasets:
        
        ##preparing the data
        X,y = fetch_data(dataset, return_X_y=True)

        train_X, test_X, train_y, test_y = train_test_split(X,y)

        ##training the models
        start = time.time()
        supportVectorMachine.svm1.fit(train_X, train_y)    
        
        ##predict, test, and create confusion matrix
        supportVectorMachine.predictf(test_X, test_y)
        
        wb = openpyxl.load_workbook(r"C:\Users\Tricide\Documents\School\EXTENDED ESSAY\Data.xlsx")
        ws = wb.active
    
        # Compute fpr, tpr, thresholds and roc auc
        fpr, tpr, thresholds = roc_curve(test_y, supportVectorMachine.probabilities[:,1])
        roc_auc = auc(fpr, tpr)
        
        # Plot ROC curve
        plt.plot(fpr, tpr, label='ROC curve (area = %0.3f)' % roc_auc)
        plt.plot([0, 1], [0, 1], 'k--')  # random predictions curve
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.0])
        plt.xlabel('False Positive Rate or (1 - Specifity)')
        plt.ylabel('True Positive Rate or (Sensitivity)')
        plt.title('Receiver Operating Characteristic')
        plt.legend(loc="lower right")
        if (supportVectorMachine.svm1.kernel == 'linear'):
            plt.savefig("ROC/{data}_{model}_{c}_roc_graph.png".format(data = dataset, model = supportVectorMachine.svm1.kernel, c = supportVectorMachine.svm1.C))
        if (supportVectorMachine.svm1.kernel == 'rbf' or supportVectorMachine.svm1.kernel == 'sigmoid'):
            plt.savefig("ROC/{data}_{model}_{c}_{g}_roc_graph.png".format(data = dataset, model = supportVectorMachine.svm1.kernel,c = supportVectorMachine.svm1.C, g =supportVectorMachine.svm1.gamma))
        if (supportVectorMachine.svm1.kernel == 'polynomial'):
            plt.savefig("ROC/{data}_{model}_{c}_{g}_{d}_roc_graph.png".format(data = dataset, model = supportVectorMachine.svm1.kernel,c = supportVectorMachine.svm1.C, g =supportVectorMachine.svm1.gamma, d = supportVectorMachine.svm1.degree))
        plt.clf()
        
        
        #Plot PR Curve
        
        precision, recall, threshold = precision_recall_curve(test_y, supportVectorMachine.probabilities[:,1])
        plt.fill_between(recall, precision)
        plt.ylabel("Precision")
        plt.xlabel("Recall")
        plt.title("Precision-Recall curve")
        if (supportVectorMachine.svm1.kernel == 'linear'):
            plt.savefig("PRC/{data}_{model}_{c}_prc_graph.png".format(data = dataset, model = supportVectorMachine.svm1.kernel, c = supportVectorMachine.svm1.C))
        if (supportVectorMachine.svm1.kernel == 'rbf' or supportVectorMachine.svm1.kernel == 'sigmoid'):
            plt.savefig("PRC/{data}_{model}_{c}_{g}_prc_graph.png".format(data = dataset, model = supportVectorMachine.svm1.kernel,c = supportVectorMachine.svm1.C, g =supportVectorMachine.svm1.gamma))
        if (supportVectorMachine.svm1.kernel == 'polynomial'):
            plt.savefig("PRC/{data}_{model}_{c}_{g}_{d}_prc_graph.png".format(data = dataset, model = supportVectorMachine.svm1.kernel,c = supportVectorMachine.svm1.C, g =supportVectorMachine.svm1.gamma, d = supportVectorMachine.svm1.degree))
        
        
        plt.clf()
        end = time.time()
        ws.append((dataset, supportVectorMachine.svm1.kernel, supportVectorMachine.svm1.C, supportVectorMachine.svm1.gamma, supportVectorMachine.svm1.degree, supportVectorMachine.tests["f1score"], "{data}_{model}_roc_graph.png".format(data = dataset, model = supportVectorMachine.svm1.kernel), "{data}_{model}_prc_graph.png".format(data = dataset, model = supportVectorMachine.svm1.kernel),supportVectorMachine.tests["accuracy"], supportVectorMachine.tests["balancedAccuracy"], supportVectorMachine.tests["specificity"], supportVectorMachine.tests["precision"], supportVectorMachine.tests["recall"], ','.join(map(str, supportVectorMachine.matrix)), end-start))
        
        wb.save("Data.xlsx")
for svm2 in supportVectorArray:
    run(svm2)            
'''
sb.boxplot(data=[[supportVectorArray[0].tests["accuracy"]], [supportVectorArray[1].tests["accuracy"]], [supportVectorArray[2].tests["accuracy"]], [supportVectorArray[3].tests["accuracy"]]], notch=True)
plt.xticks([0, 1,2,3], ['RBFSVM', 'PolynomialSVM', 'LinearSVM', 'SigmoidSVM'])
plt.ylabel('Test Accuracy')
plt.savefig("ModelComparison/{}_ModelComparisons.png".format(dataset))
'''
    
    #representation of hyperplane and data
    
    
    