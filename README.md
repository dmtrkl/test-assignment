1. For which tasks is it better to use Logistic Regression instead of other models?
    Logistic Regression is suitable for classification task when number of samples is much greater then number of features, target variable is binary and desicions should be supported with a probability for each class.

2. What are the most important parameters that you can tune in Linear Regression / Logistic Regression / SVM?
    Regularization parameters.

3. How does parameter C influence regularisation in Logistic Regression?
    C is an inverse to the lambda coeficient in the regularization term. Smaller the value, more restrictions is applied to the model complexity.

4. Which top 3 features are the most important for each data sets?
    For the Heart-Disease dataset Linear model with standartized input assigns highest coeficients to: "sex", "ca", "fbs", while for tree-based methods "cp", "exang", "chol" as the most important features. EDA shows high correlation between features, so these results can change significantly within small perturbation of the inputs. For the Medical-Cost dataset: "smoker", "region", "children" have high coeficients in the linear model and "age", "smoker", "bmi" are important for the tree-based methods.
    
5. What accuracy metrics did you receive on train and test sets for `Heart Disease UCI` dataset?
    Train and test accuracy averaged across 5 Fold Cross-Validation equals 0.839 and 0.792 accordingly
 
6. What MSE did you receive on train and test datasets for `Medical Cost Personal`?
    Train and test MSE averaged across 5 Fold Cross-Validation equals 6026^2 and 6068^2 accordingly
