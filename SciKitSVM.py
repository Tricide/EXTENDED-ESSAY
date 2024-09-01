from pmlb import fetch_data, classification_dataset_names
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, f1_score, precision_score, balanced_accuracy_score, roc_auc_score, roc_curve, auc, precision_recall_curve

import matplotlib.pyplot as plt
import seaborn as sb
import openpyxl
##for ease of use
class supportVector:
    
    def __init__(self, kernel_type):
        self.svm1 = svm.SVC(kernel = kernel_type, cache_size = 1000, probability = True)
        self.matrix = []
        self.predictions = []
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
        print(self.svm1.kernel)
        print(self.predictions)
        self.matrix = confusion_matrix(y, self.predictions).ravel()
        self.tests["f1score"] = f1_score(y, self.predictions)
        self.tests["accuracy"] = accuracy_score(y, self.predictions)
        self.tests["balancedAccuracy"] = balanced_accuracy_score(y, self.predictions)
        self.tests["precision"] = precision_score(y,self.predictions)
        ##self.tests["specificity"] = recall_score(y, self.predictions, pos_label = 0)
        self.tests["recall"] = recall_score(y, self.predictions)
        self.tests["rocauc"] = roc_auc_score(y, self.predictions)
    
        
        
datasets = ["banana"]        


for dataset in datasets:
            
    ##initiating all necessary variables 
    rbf_svm = supportVector('rbf')
    polynomial_svm = supportVector('poly')
    linear_svm = supportVector('linear')
    sig_svm = supportVector('sigmoid')

    supportVectorArray = [rbf_svm, polynomial_svm, linear_svm, sig_svm]

    ##preparing the data
    X,y = fetch_data(dataset, return_X_y=True)

    train_X, test_X, train_y, test_y = train_test_split(X,y)

    ##training the models
    for svmachine in supportVectorArray:
        svmachine.svm1.fit(train_X, train_y)    
        
        ##predict, test, and create confusion matrix
        svmachine.predictf(test_X, test_y)
    
    wb = openpyxl.load_workbook(r"C:\Users\Tricide\Documents\School\EXTENDED ESSAY\Data.xlsx")
    ws = wb.active
    for svmachine in supportVectorArray:
        # Compute fpr, tpr, thresholds and roc auc
        fpr, tpr, thresholds = roc_curve(test_y, svmachine.probabilities[:,1])
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
        plt.savefig("{data}_{model}_roc_graph.png".format(data = dataset, model = svmachine.svm1.kernel))
        plt.clf()
        
        
        #Plot PR Curve
        
        precision, recall, threshold = precision_recall_curve(test_y, svmachine.probabilities[:,1])
        plt.fill_between(recall, precision)
        plt.ylabel("Precision")
        plt.xlabel("Recall")
        plt.title("Precision-Recall curve")
        plt.savefig("{data}_{model}_prc_graph.png".format(data = dataset, model = svmachine.svm1.kernel))
        
        
        plt.clf()
        
        ws.append((dataset, svmachine.svm1.kernel, svmachine.tests["f1score"], "{data}_{model}_roc_graph.png".format(data = dataset, model = svmachine.svm1.kernel), "{data}_{model}_prc_graph.png".format(data = dataset, model = svmachine.svm1.kernel),svmachine.tests["accuracy"], svmachine.tests["balancedAccuracy"], svmachine.tests["specificity"], svmachine.tests["precision"], svmachine.tests["recall"], ','.join(map(str, svmachine.matrix))))
        
    wb.save("Data.xlsx")
    sb.boxplot(data=[[supportVectorArray[0].tests["accuracy"]], [supportVectorArray[1].tests["accuracy"]], [supportVectorArray[2].tests["accuracy"]], [supportVectorArray[3].tests["accuracy"]]], notch=True)
    plt.xticks([0, 1,2,3], ['RBFSVM', 'PolynomialSVM', 'LinearSVM', 'SigmoidSVM'])
    plt.ylabel('Test Accuracy')
    plt.savefig("{}_ModelComparisons.png".format(dataset))