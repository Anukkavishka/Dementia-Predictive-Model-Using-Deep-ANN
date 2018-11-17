# -*- coding: utf-8 -*-
"""
Created on Sun Jan  7 21:16:14 2018

@author: Administrator
"""

from flask import Flask, json, jsonify, request,Response
import pandas as pd
import numpy as np
from imblearn.over_sampling import SMOTE
from keras.models import model_from_json
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.wrappers.scikit_learn import KerasClassifier
import pickle
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.externals import joblib
from sklearn.model_selection import cross_val_score

app = Flask(__name__)

columns = ['type','amount','oldbalanceOrg','newbalanceOrig','oldbalanceDest','newbalanceDest','isFraud']
dependent_variable=columns[-1]

def build_classifier2():
    classifier = Sequential()
    classifier.add(Dense(units = 5, kernel_initializer = 'uniform', activation = 'relu', input_dim = 11))
    classifier.add(Dropout(rate = 0.1))
    classifier.add(Dense(units = 3, kernel_initializer = 'uniform', activation = 'relu'))
    classifier.add(Dropout(rate = 0.1))
    classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))
    classifier.compile(optimizer = 'rmsprop', loss = 'binary_crossentropy', metrics = ['accuracy'])
    return classifier

def build_classifier(optimizer):
    classifier = Sequential()
    classifier.add(Dense(units = 5, kernel_initializer = 'uniform', activation = 'relu', input_dim = 11))
    classifier.add(Dropout(rate = 0.1))
    classifier.add(Dense(units = 3, kernel_initializer = 'uniform', activation = 'relu'))
    classifier.add(Dropout(rate = 0.1))
    classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))
    classifier.compile(optimizer = optimizer, loss = 'binary_crossentropy', metrics = ['accuracy'])
    return classifier

def load_trained_model(name):
    json_file = open('model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    loaded_model.load_weights(name)
    print("Model loaded...")
    return loaded_model

def load_scaler(name):
    json_file = open('model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    loaded_model.load_weights(name)
    print("Model loaded...")
    return loaded_model


################################################################################################################
@app.route('/ANN/detect', methods=["POST"])
def check_for_frauds():    
   json_ = request.json   
   dataset = pd.DataFrame(json_, index=[0]) 
   dataset = dataset[dataset.columns.difference([dependent_variable])] 
    
   #dataset=pd.get_dummies(dataset)   
   X = dataset.reindex(columns = model_columns, fill_value=0)#fill_value=0 null->0
   
   #test_scaled_rec = scaler.transform(X)
   print(test_scaled_rec); 
     
   score = model.predict(test_scaled_rec)
   isFraud = (score > 0.05)
   data = {
        'isFraud'  : np.bool(isFraud),
        'scorre' : np.float64(score)
   }
   js = json.dumps(data)

   response = Response(js, status=200, mimetype='application/json')
   return response
################################################################################################################
@app.route('/ANN/train', methods=["POST"])
def train_model():  
    
    json_ = request.json 
    split_size = np.float64(json_['test_size'])
    batch_size = np.int64(json_['batch_size'])
    epochs = np.int64(json_['epochs'])    
    optimizer = json_['optimizer']
    
    dataset = pd.read_csv('Data4.csv')
    dataset = dataset[columns]
    
    
    categoricals = []  
    for col, col_type in dataset.dtypes.iteritems():
        if col_type == 'O':
            categoricals.append(col)
        else:
            dataset[col].fillna(0, inplace=True) 
    
    ohe_dataset = pd.get_dummies(dataset, columns=categoricals, dummy_na=True)
    X = ohe_dataset[ohe_dataset.columns.difference([dependent_variable])]
    Y = ohe_dataset[dependent_variable]
    
    from sklearn.model_selection import train_test_split
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = split_size, random_state = 0)
    
    sm = SMOTE(random_state=12, ratio = 0.9)
    X_train, Y_train = sm.fit_sample(X_train, Y_train)

    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)
    
    classifier=build_classifier(optimizer)
    classifier.fit(X_train, Y_train, batch_size = batch_size, epochs = epochs)
    
  
    Y_pred = classifier.predict(X_test)
    Y_pred = (Y_pred > 0.05)
    
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(Y_test, Y_pred)
    
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report
    
    accuracy_score=accuracy_score(Y_test,Y_pred)
    f1_score=f1_score(Y_test, Y_pred, average="macro")
    precision_score=precision_score(Y_test, Y_pred, average="macro")
    recall_score=recall_score(Y_test, Y_pred, average="macro")

    print ('\n clasification report:\n', classification_report(Y_test, Y_pred))
    
    model_json = classifier.to_json()
    with open("model.json", "w") as json_file:
        json_file.write(model_json)
    classifier.save_weights("model.h5")
    print("Saved model to disk")    
    
    import pickle
    pickle.dump(sc, open('scaler.sav', 'wb'))    
    
    from sklearn.externals import joblib
    model_columns = list(X.columns)
    joblib.dump(model_columns, 'model_columns.pkl')    

    data = {   
        'accuracy_score' : np.float64(accuracy_score),
        'f1_score' : np.float64(f1_score),
        'precision_score': np.float64(precision_score),
        'recall_score'  : np.float64(recall_score)
    }
    js = json.dumps(data)

    response = Response(js, status=200, mimetype='application/json')
    return response

################################################################################################################ 
@app.route('/ANN/evaluate', methods=["POST"])
def evaluate_model():  
    
    json_ = request.json 
    split_size = np.float64(json_['test_size'])
    batch_size = np.int64(json_['batch_size'])
    epochs = np.int64(json_['epochs']) 
    cross_validation = np.int64(json_['cross_validation'])    
    
    dataset = pd.read_csv('Data4.csv')
    dataset = dataset[columns]
    
    
    categoricals = []  
    for col, col_type in dataset.dtypes.iteritems():
        if col_type == 'O':
            categoricals.append(col)
        else:
            dataset[col].fillna(0, inplace=True) 
    
    ohe_dataset = pd.get_dummies(dataset, columns=categoricals, dummy_na=True)
    X = ohe_dataset[ohe_dataset.columns.difference([dependent_variable])]
    Y = ohe_dataset[dependent_variable]
    
    from sklearn.model_selection import train_test_split
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = split_size, random_state = 0)
    
    sm = SMOTE(random_state=12, ratio = 0.9)
    X_train, Y_train = sm.fit_sample(X_train, Y_train)
    
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)
    classifier = KerasClassifier(build_fn = build_classifier2, batch_size = batch_size, epochs = epochs)
    accuracies = cross_val_score(estimator = classifier, X = X_train, y = Y_train, cv = cross_validation)
    mean = accuracies.mean()
    variance = accuracies.std()

    data = {   
        'mean' : np.float64(mean),
        'variance' : np.float64(variance)
    }
    js = json.dumps(data)

    response = Response(js, status=200, mimetype='application/json')
    return response

################################################################################################################
if __name__ == "__main__":
        model_columns = joblib.load('model_columns.pkl')
        model = load_trained_model("model.h5")
        scaler = pickle.load(open('scaler.sav', 'rb'))
        app.run()