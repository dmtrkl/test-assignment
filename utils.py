from numpy import NaN
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import os, sys
from itertools import takewhile

def get_attr_names(description_path, attr_num=14):
    """Obtain feature names for Heart Disease data set"""
    with open(description_path) as f:
        description = f.readlines()
    idx = 0
    for i, line in enumerate(description):
        if 'Attribute Information' in line:
            idx = i + 2
    attr_names = description[idx:(idx + attr_num)]
    if not idx:
        raise Exception('Error while searching for attribute names')
    else:
        return list(map(lambda s: s[s.find("(")+1:s.find(")")], attr_names))  
    
    
def read_data():
    """Merge four data sources for Heart Disease data set"""
    attr_names = get_attr_names(os.path.join('data', 'heart-disease.names'))
    dataset_names = [
        'cleveland',
        'va',
        'switzerland',
        'hungarian'
    ]
    
    dataset = {
        name:pd.read_csv(
            os.path.join('data', f'processed.{name}.data'),
            names=attr_names
        )
        for name in dataset_names
    }
    
    return pd.concat(dataset.values(), ignore_index=True)


def prepare_data(dataset_name):
    """Preprocess data before training linear models"""
    if dataset_name == 'heart-disease':
        data = read_data()
        categorical_features = [
            'sex',
            'cp',
            'fbs',
            'restecg',
            'exang',
            'slope',
            'thal',
            'ca'
        ]
        target = ['num']
        numeric_features = list(set(data.columns) - set(categorical_features) - set(target))
        features = list(set(data.columns) - set(target))
        data.replace('?', NaN, inplace=True)
        data.num = data.num.apply(lambda x: 1 if x > 0 else 0)
        data = data.astype(float)
        for feature in numeric_features:
            data[feature].fillna(value=data[feature].median(), inplace=True)  
        X = pd.get_dummies(
                data.drop('num', axis=1),
                columns=categorical_features,
                dummy_na=True,
                drop_first=True,
                dtype='float'
        )
        #Remove redundant nan indicator columns
        X.drop((X.sum()[X.sum() < 5]).index.to_list(), axis=1, inplace=True)
        scaler = StandardScaler()
        X[numeric_features] = scaler.fit_transform(X[numeric_features])  
        
        y = data['num']
        return X.columns, X.to_numpy(), y.to_numpy()  
        
    elif dataset_name == 'medical-cost':
        data = pd.read_csv(os.path.join('data', 'insurance.csv'))
        categorical_features = ['sex', 'smoker', 'region', 'children']
        target = ['charges']
        numeric_features  = list(set(data.columns) - set(categorical_features) - set(target))
        for f in numeric_features:
            data[f] = data[f].astype(float)
        X = pd.get_dummies(
            data.drop('charges', axis=1),
            columns=categorical_features,
            drop_first=True,
            dtype='float'
        )
        y = data[target].astype(float)
        return X.columns, X.to_numpy(), y.to_numpy()
    else:
        raise ValueError('Unknown data set name')

def eda_heart_disease():
    data = read_data()
    # Process data to perform auto EDA
    data.replace('?', NaN, inplace=True)
    for f in data.columns:
        # Tackle mixed types and suppress variables into optimal representation that support NaN
        data[f] = pd.to_numeric(data[f]).convert_dtypes()
        # Force boolean type instead Int64 (https://github.com/pandas-dev/pandas/issues/32287)
        if len(data[f].unique()) == 2 or f == 'num':
            data[f]= data[f].apply(lambda x: True if x > 0 else False) 
    # Generate HTML report and show brief info
    config = {'cat': {'words':False, 'characters':False}}
    data.profile_report(
        vars=config,
        title="Heart Disease data set",
        dark_mode=True
    ).to_file('EDA-heart-disease.html')
    return data

def eda_medical_cost():
    data = pd.read_csv(os.path.join('data', 'insurance.csv'))
    data = data.convert_dtypes()
    config = {'cat': {'words':False, 'characters':False}}
    data.profile_report(
        vars=config,
        title="Medical cost data set",
        dark_mode=True
    ).to_file('EDA-medical-cost.html')
    
def top3features(model, col_names):
    if getattr(model, 'feature_importances_', None) is not None:
        importance = getattr(model, 'feature_importances_')
    elif getattr(model, 'coef_', None) is not None:
        importance = getattr(model, 'coef_')[0]
    else:
        importance = getattr(model, 'weights')[1:]
    features = dict(zip(col_names, importance))
    importance = sorted(features, key=lambda key: abs(features[key]), reverse=True)
    base_features = [f.split('_')[0] if '_' in f else f for f in importance]
    seen = set()
    top3 = set(takewhile(lambda x: seen.add(x) or len(seen) <= 3, base_features ))
    return top3
    
class HidePrint:
    def __enter__(self):
        self._stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._stdout
    