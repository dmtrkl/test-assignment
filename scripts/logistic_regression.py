import os, sys
parentddir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir))
sys.path.append(parentddir)

from collections import defaultdict
from models.linear_models import LogReg
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn.metrics import RocCurveDisplay
from sklearn.metrics import accuracy_score 
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
from utils import prepare_data, HidePrint, top3features

def evaluate(model, X, y, n_splits=5, **kwargs):
    metrics = defaultdict(list)
    train_acc = list()
    cv = KFold(n_splits=n_splits)    
    with HidePrint():
        print('Test model..')
        for i, (train, test) in enumerate(cv.split(X)):
            model.fit(X[train], y[train])
            pred = model.predict(X[test], **kwargs)
            train_acc.append(
                accuracy_score(
                    y[train],
                    model.predict(X[train], **kwargs)
                )
            )
            metrics['Accuracy'].append(accuracy_score(y[test], pred))
            metrics['Precision'].append(precision_score(y[test], pred))
            metrics['Sensitivity'].append(recall_score(y[test], pred))
            metrics['F1'].append(f1_score(y[test], pred))
            prob = model.predict_proba(X[test])[:, 1]
            metrics['ROC_AUC'].append(roc_auc_score(y[test], prob))
            if i == 0:
                fpr, tpr, thresholds = roc_curve(y[test], prob)
                roc_auc = auc(fpr, tpr)
                display = RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc)
                display.plot()
                plt.title('ROC curve (1-st CV split)')
                plt.show()
                
    print(f'({n_splits} Fold) Cross-Validation\n')
    print(f'Train Accuracy: {sum(train_acc)/len(train_acc)}\n')
    for name, values in metrics.items():
        print(f'Test {name}: {sum(values)/ len(values)}')

def main():
    col_names, X, y = prepare_data(dataset_name='heart-disease') 
    model = LogReg()
    model.fit(X, y)
    evaluate(LogReg(), X, y)
    print(f'\nTop3 feature according to {model} : ' + ', '.join(top3features(model, col_names)))
    return model

if __name__ == '__main__':
    main()