import os, sys
parentddir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir))
sys.path.append(parentddir)

import argparse
from numpy import sqrt, log, exp
from models.linear_models import LinReg
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error as mse
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from utils import HidePrint, prepare_data, top3features

        
def evaluate(model, X, y, n_splits=10, **kwargs):
    with HidePrint():
        train_rmse = list()
        results = list()
        cv = KFold(n_splits=n_splits)
        for i, (train, test) in enumerate(cv.split(X)):
            model.fit(X[train], y[train], **kwargs)
            train_rmse.append(sqrt(mse(y[train], model.predict(X[train], **kwargs))))
            pred = model.predict(X[test], **kwargs)
            rmse = sqrt(mse(y[test], pred))
            results.append(rmse)
            if i == 0:
                max_val = max(int(max(y[test])), int(max(pred)))
                plt.scatter(y[test], pred)
                plt.plot([0, max_val], [0, max_val], '--k')  
                plt.xlabel('Test target')
                plt.ylabel('Predicted target')
                plt.title('Predictions VS True values (1-st CV split)')
                plt.show()
    print(f"({n_splits} Fold) Cross-validation\n")
    print(f'Train RMSE: {sum(train_rmse)/ len(train_rmse)}')
    print(f'Test RMSE: {sum(results)/ len(results)}')
    

def main(*args, **kwargs):
    col_names, X, y = prepare_data(dataset_name='medical-cost')
    model = LinReg(reg_lambda=kwargs.get('lambda') or 0)
    model.fit(X, y)
    results = evaluate(LinReg(), X, y, *args, **kwargs)
    print(f'\nTop3 feature according to {model} : ' + ', '.join(top3features(model, col_names)))
    return model
   
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-t",
                        "--log-transform",
                        help=("Log-transform target variable before training"
                              "and take exponent of the output predictions"),
                        action="store_true")
    parser.add_argument("-l",
                        "--lambda",
                        type=float,
                        help="L2 regularization strength")
    args = parser.parse_args()
    main(**vars(args))
    

