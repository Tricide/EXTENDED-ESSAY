from pmlb import fetch_data, classification_dataset_names
from sklearn import svm
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, f1_score, precision_score, balanced_accuracy_score, roc_auc_score, roc_curve, auc, precision_recall_curve
from sklearn.datasets import make_classification
from sklearn import preprocessing
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
import openpyxl
import time
##for ease of use
class supportVector:
    def __init__(self, kernel_type, c, gamma, degree, shift):
        self.svm1 = svm.SVC(kernel = kernel_type, cache_size = 1000, probability = True, C=c, gamma = gamma, degree = degree, coef0 = shift)
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
        self.tests["specificity"] = self.matrix[0]/(self.matrix[0]+self.matrix[1])
        self.tests["precision"] = precision_score(y,self.predictions)
        self.tests["recall"] = recall_score(y, self.predictions)
        self.tests["rocauc"] = roc_auc_score(y, self.predictions)

#Creates a PRC Graph            
def plotPRCGraph(svm2, dataset, test_y):
    
    precision, recall, threshold = precision_recall_curve(test_y, svm2.probabilities[:,1])
    plt.fill_between(recall, precision)
    plt.ylabel("Precision")
    plt.xlabel("Recall")
    plt.title("Precision-Recall curve")
    if (svm2.svm1.kernel == 'linear'):
        plt.savefig("PRC/{data}_{model}_{c}_prc_graph.png".format(data = dataset, model = svm2.svm1.kernel, c = svm2.svm1.C))
    if (svm2.svm1.kernel == 'rbf' or svm2.svm1.kernel == 'sigmoid'):
        plt.savefig("PRC/{data}_{model}_{c}_{g}_prc_graph.png".format(data = dataset, model = svm2.svm1.kernel,c = svm2.svm1.C, g =svm2.svm1.gamma))
    if (svm2.svm1.kernel == 'poly'):
        plt.savefig("PRC/{data}_{model}_{c}_{g}_{d}_prc_graph.png".format(data = dataset, model = svm2.svm1.kernel,c = svm2.svm1.C, g =svm2.svm1.gamma, d = svm2.svm1.degree))
    plt.clf()

#Creates a ROC Graph            
def plotROCGraph(svm2, dataset, test_y):
    # Compute fpr, tpr, thresholds and roc auc
    fpr, tpr, thresholds = roc_curve(test_y, svm2.probabilities[:,1])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label='ROC curve (area = %0.3f)' % roc_auc)
    plt.plot([0, 1], [0, 1], 'k--')  # random predictions curve
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Positive Rate or (1 - Specifity)')
    plt.ylabel('True Positive Rate or (Sensitivity)')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    if (svm2.svm1.kernel == 'linear'):
        plt.savefig("ROC/{data}_{model}_{c}_roc_graph.png".format(data = dataset, model = svm2.svm1.kernel, c = svm2.svm1.C))
    if (svm2.svm1.kernel == 'rbf' or svm2.svm1.kernel == 'sigmoid'):
        plt.savefig("ROC/{data}_{model}_{c}_{g}_roc_graph.png".format(data = dataset, model = svm2.svm1.kernel,c = svm2.svm1.C, g =svm2.svm1.gamma))
    if (svm2.svm1.kernel == 'poly'):
        plt.savefig("ROC/{data}_{model}_{c}_{g}_{d}_roc_graph.png".format(data = dataset, model = svm2.svm1.kernel,c = svm2.svm1.C, g =svm2.svm1.gamma, d = svm2.svm1.degree))
    plt.clf()
    return roc_auc

#functions for PlotDecisionSurface
def make_meshgrid(x, y, h=0.2):
    x_min, x_max = x.min() - 1, x.max() + 1
    y_min, y_max = y.min() - 1, y.max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    return xx, yy

#function for PlotDecisionSurface
def plot_contours(ax, clf, xx, yy, **params):
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    out = ax.contourf(xx, yy, Z, **params)
    return out

#plots SVM Decision Surface
def plotDecisionSurface(X, estimator, y, svm2, dataset):
    ###plot the decision surface
    fig, ax = plt.subplots()
    # title for the plots
    title = ('Decision surface of SVM')
    # Set-up grid for plotting.
    X0, X1 = X[:, 0], X[:, 1]
    xx, yy = make_meshgrid(X0, X1)
    plot_contours(ax, estimator, xx, yy, cmap=plt.cm.coolwarm, alpha=0.8)
    ax.scatter(X0, X1, c=y, cmap=plt.cm.coolwarm, s=20, edgecolors='k')
    ax.set_ylabel('Feature 1')
    ax.set_xlabel('Feature 2')
    ax.set_xticks(())
    ax.set_yticks(())
    ax.set_title(title)
    ax.legend()
    if (svm2.svm1.kernel == 'linear'):
        plt.savefig("DecisionBoundary/{data}_{model}_{c}_DecisionBoundary_graph.png".format(data = dataset, model = svm2.svm1.kernel, c = svm2.svm1.C))
    if (svm2.svm1.kernel == 'rbf' or svm2.svm1.kernel == 'sigmoid'):
        plt.savefig("DecisionBoundary/{data}_{model}_{c}_{g}_DecisionBoundary_graph.png".format(data = dataset, model = svm2.svm1.kernel,c = svm2.svm1.C, g =svm2.svm1.gamma))
    if (svm2.svm1.kernel == 'poly'):
        plt.savefig("DecisionBoundary/{data}_{model}_{c}_{g}_{d}_DecisionBoundary_graph.png".format(data = dataset, model = svm2.svm1.kernel,c = svm2.svm1.C, g =svm2.svm1.gamma, d = svm2.svm1.degree))
    plt.clf()

#run model with preset dataset and support vector hyperparameters    
def run(supportVectorMachine, X, y, dataset):            
    
    ##preparing the data
    train_X, test_X, train_y, test_y = train_test_split(X,y)

    ##training the models and record training time
    start = time.time()
    estimator = supportVectorMachine.svm1.fit(train_X, train_y)    
    end = time.time()
    
    ##predict, test, and create confusion matrix
    supportVectorMachine.predictf(test_X, test_y)
    
    # Plot ROC curve and calculate area under curve
    roc_auc = plotROCGraph(supportVectorMachine, dataset, test_y)
    
    #Plot PR Curve
    plotPRCGraph(supportVectorMachine, dataset, test_y)
    
    #Plot Decision Surface
    #if (X.ndim == 2):
    #    plotDecisionSurface(X, estimator, y, supportVectorMachine, dataset)
    
    #open excel sheet
    wb = openpyxl.load_workbook(r"C:\Users\Tricide\Documents\School\EXTENDED ESSAY\Data.xlsx")
    ws = wb.active
    ##append data to excel sheet
    ws.append((dataset, supportVectorMachine.svm1.kernel, supportVectorMachine.svm1.C, supportVectorMachine.svm1.gamma, supportVectorMachine.svm1.degree, supportVectorMachine.svm1.coef0, supportVectorMachine.tests["f1score"], roc_auc, "{data}_{model}_roc_graph.png".format(data = dataset, model = supportVectorMachine.svm1.kernel), "{data}_{model}_prc_graph.png".format(data = dataset, model = supportVectorMachine.svm1.kernel),supportVectorMachine.tests["accuracy"], supportVectorMachine.tests["balancedAccuracy"], supportVectorMachine.tests["specificity"], supportVectorMachine.tests["precision"], supportVectorMachine.tests["recall"], ','.join(map(str, supportVectorMachine.matrix)), end-start))
    
    wb.save("Data.xlsx")
    plt.close('all')
    
##all values to be tested        
datasets = ["banana", "clean2", "parity5",  "GAMETES_Epistasis_2_Way_1000atts_0.4H_EDM_1_EDM_1_1"]       
sampleAmounts = [100, 1000, 10000]
features  = [1000]
informativeFeatures = [.1,.25, .5, .75, 1] ##probably will not be used
gammaValues = [0.001, 0.01, 0.1, 1, 10]
cValues = [0.001, 0.01, 0.1, 1, 10]
degreeValues = [2,3,4]
shiftValues = [0.001, 0.01, 0.1, 1, 10]


for feature in features:
    for amount in sampleAmounts:
        randomState = np.random.random_integers(0,1000)
        X_in, y_in = make_classification(n_samples = amount, n_features = feature, n_informative = feature, n_redundant = 0)
        dataset = "synthetic_{amount}_{features}".format(amount = amount, features = feature)
        print("{num} samples in {state} randomState".format(num = amount, state = randomState))
        for c in cValues:
            linear_svm = supportVector('linear', c, 0, 3, 0)
            run(linear_svm, X_in, y_in, dataset)
            
            for gamma in gammaValues:
                rbf_svm = supportVector('rbf', c, gamma, 3 ,0)
                run(rbf_svm, X_in, y_in, dataset)
                
                for shift in shiftValues:
                    sig_svm = supportVector('sigmoid', c, gamma, 3, shift)
                    run(sig_svm, X_in, y_in, dataset)
                    
            #for degree in degreeValues:
            #    polynomial_svm = supportVector('poly', c, 1, degree, 0)
            #    run(polynomial_svm, X_in, y_in, dataset)

for dataset in datasets:
    x_in, y_in = fetch_data(dataset, return_X_y=True)
    for c in cValues:
            linear_svm = supportVector('linear', c, 0, 3, 0)
            run(linear_svm, X_in, y_in, dataset)
            
            for gamma in gammaValues:
                rbf_svm = supportVector('rbf', c, gamma, 3 ,0)
                run(rbf_svm, X_in, y_in, dataset)
                
                for shift in shiftValues:
                    sig_svm = supportVector('sigmoid', c, gamma, 3, shift)
                    run(sig_svm, X_in, y_in, dataset)
                    
            for degree in degreeValues:
                polynomial_svm = supportVector('poly', c, 1, degree, 0)
                run(polynomial_svm, X_in, y_in, dataset)


##X_in,y_in = fetch_data("banana", return_X_y=True)
