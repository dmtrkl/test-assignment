import numpy as np 
import numpy as np

class LogReg():
    """Logistic regression using gradient descent of cross entropy"""
    
    def __init__(self):
        self.weights = None
        self.features = None
    
    def __str__(self):
        return 'LogReg()'

    def init_bias(self, features):
        """Switch to homogeneous coordinates.
        Add feature that is constantly equals to 1.
        """
        return np.hstack((np.ones((features.shape[0], 1)), features))
                                  
    def log_loss(self, target):
        """Calculate Log Loss"""
        scores = np.dot(self.features, self.weights)
        ll = target * scores - np.log(1 + np.exp(scores))
        ll = - sum(ll)/len(ll)
        
        return ll

    def grad(self, target, predictions):
        """Compute gradient"""
        output_error_signal = target - predictions
        gradient = np.dot(self.features.T, output_error_signal)
        
        return gradient

    def sigmoid(self, x):
        """Apply activation to obtain probabilities"""
        return 1 / (1+np.exp(-x))

    def fit(self, features, target, num_steps=2000, learning_rate=6e-3):
        """Update model weights with respect to the gradient and the learning rate"""
        self.features = self.init_bias(features)
        self.weights = np.zeros(self.features.shape[1])
        
        iter_num = []
        train_score = []
        print('Training starts..')
        
        for num,step in enumerate(range(num_steps + 1)):

            scores = np.dot(self.features, self.weights)
            predictions = self.sigmoid(scores)

            gradient = self.grad(target, predictions)
            self.weights += learning_rate * gradient

            log_loss_train = self.log_loss(target)
            
            train_score.append(log_loss_train)
            iter_num.append(num)
            
            if step % 200 == 0  :
                print(f'Iteration: {step}\tTrain loss: {log_loss_train}')
                
        print('Training finished successfully!\n')
        self.loss_hist = [iter_num, train_score]

    def predict(self, batch):
        """Threshold probabilties to predict class.
        Can be later optimized for a specific metric
        """
        return (self.predict_proba(batch)[:, 1] > 0.5)
        
    def predict_proba(self, batch):
        """Generate output probabilties for both target classes"""
        X = self.init_bias(batch)
        score = self.sigmoid(np.dot(X, self.weights))
        prob = score.reshape(score.shape[0], 1)
        prob = np.insert(prob, 0, 1 - prob.flatten(), axis=1) 
        
        return prob
        
class LinReg():
    """Closed-form solution for multivariate OLS 
    
    Parameters
    ----------
    reg_lambda : `float`, optional
       L2 regularization strength
    """
 
    def __init__(self, reg_lambda=0):
        self.reg_lambda = reg_lambda
        
    def __str__(self):
        return f'LinReg(reg_lambda={self.reg_lambda})'       
    
    def init_bias(self, features):
        """Switch to homogeneous coordinates.
        Add feature that is constantly equals to 1.
        """        
        return np.insert(features, 0, 1, axis=1)
        
    def fit(self, features, target, **kwargs):
        """Compute model weights"""
        X, y = np.array(features), np.array(target)
        if kwargs.get('log_transform'):
            y = np.log(y)
        X = self.init_bias(X)
        print('Training starts..')
        xTx = X.T @ X
        self.weights = np.linalg.inv(xTx + self.reg_lambda*np.identity(xTx.shape[0])) @ X.T @ y
        print('Training finished successfully!\n')
            
    def predict(self, features, **kwargs):
        """Generate estimates for the target variable"""
        features = np.array(features)
        features = self.init_bias(features)
        pred = features @ self.weights
        if kwargs.get('log_transform'):
            pred = np.exp(pred)
        return pred

        
