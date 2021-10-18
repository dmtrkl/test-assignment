from models.linear_models import LogReg, LinReg
import numpy as np
np.set_printoptions(suppress=True)
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from unittest import TestCase, main
from utils import prepare_data


class TestWeights(TestCase):
    
    @classmethod
    def setUpClass(cls):
        cls.data = {name : prepare_data(name)[1:] for name in ('heart-disease', 'medical-cost')}
    
    def test_logistic_regression(self):
        X, y = self.data['heart-disease']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
        LR = LogReg()
        LR.fit(X_train, y_train)  
        LR_ref = LogisticRegression(penalty='none', max_iter=2000)
        LR_ref.fit(X_train, y_train)  
        self.assertIsNone(
            np.testing.assert_almost_equal(
                LR.weights,
                np.append(LR_ref.intercept_, LR_ref.coef_),
                decimal=3
            )
        )                
           
    def test_ols_regression(self):
        X, y = self.data['medical-cost']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
        LR = LinReg()
        LR.fit(X_train, y_train)
        LR_ref = LinearRegression()
        LR_ref.fit(X_train, y_train)  
        self.assertIsNone(
            np.testing.assert_almost_equal(
                LR.weights[:, 0],
                np.append(LR_ref.intercept_, LR_ref.coef_),
                decimal=3
            )
        )
                     
    def test_ridge_regression(self):
        X, y = self.data['medical-cost']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
        LR = LinReg(reg_lambda=1)
        LR.fit(X_train, y_train)
        LR_ref = Ridge(solver="cholesky", tol=1e-6, alpha=1)
        LR_ref.fit(X_train, y_train) 
        self.assertIsNone(
            np.testing.assert_almost_equal(
                LR.weights[:, 0],
                np.append(LR_ref.intercept_, LR_ref.coef_),
                decimal=3
            )
        )
        
if __name__ == "__main__":
    main()