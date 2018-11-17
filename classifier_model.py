#making the data mapped in a more complex pattern
from sklearn.datasets import make_circles
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
from tensorflow.python.keras.models import model_from_json
from tensorflow.python.keras.preprocessing.text import Tokenizer
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder
from flask import Flask, json, jsonify, request,Response
import requests

app = Flask(__name__)
#get the training data
original_training=pd.read_csv(".\\ADvsHCFourier.csv")
original_training.head(10)
X_train=original_training.copy()


del X_train['class']
del X_train['experiment']
X_train.head()
#get the columns of the data set from the data frame
#print(X_train.columns)
#print("print the training data X : \n",X_train)


y_train = []

for data in original_training ['class']:
    y_train.append(data)
#below code encode the classes ['AD','HCI'] into [0,1]
encoder = LabelEncoder()
encoder.fit(y_train)
Y_train = encoder.transform(y_train)
#print("print train data y values  :",Y_train)    

#getting the test dataset
test_data= pd.read_csv(".\\ADvsHCFourier_test.csv")
X_test=test_data.copy()
del X_test['class']
del X_test['experiment']
X_test.head(8)
#print("print test X data : \n",X_test)


y_test=[]
for t_data in test_data ['class']:
    y_test.append(t_data)

encoder1 = LabelEncoder()
encoder1.fit(y_test)
Y_test = encoder.transform(y_test)
    
#print("print test data y values :",Y_test)    
################################################################################################################

def load_trained_model(name):
    json_file = open('model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    loaded_model.load_weights(name)
    print("Model loaded...")
    return loaded_model


#################################################################################################################
#creating the model and compiling
from sklearn.naive_bayes import MultinomialNB
clf=MultinomialNB()
clf.fit(X_train.abs(),y_train)    
   
#################################################################################################################
#training and evalutaing the model
print('predicted labels by the model')
print(clf.predict(X_test))      
        ##Evaluate Performance##

from sklearn.metrics import accuracy_score
accuracy_score(Y_test, clf.predict(X_test))*100
    
print("\n\nAccuracy Score : ",accuracy_score)

import pickle
# save the model to disk

pickle.dump(clf ,open('finalized_model.sav', 'wb'))


# load the model from disk
#result = loaded_model.score(X_test, Y_test)
#print(result)
#################################################################################################################

################################################################################################################
@app.route('/detect/', methods=["POST"])
def check_for_dementia():    
   json_ = request.json 

   # print(json_,"\n\n\n")  
   dataset = pd.DataFrame(json_, index=[0]) 
   #print("Dataset : \n\n",dataset)
     
   score = loaded_model.predict(dataset)
   #print("\n",score,"\n")
   #score = 0.09
   #isDaignosed = (score > 0.05)
   print("\n The score from the loaded model:",score)

   data = {
        'isDaignosed'  : "True",
        'scorre' : score# np.float64(score)
   }

   js = json.dumps(data)

   response = Response(js, status=200, mimetype='application/json')
   return response
################################################################################################################
if __name__ == "__main__":
        #model_columns = joblib.load('model_columns.pkl')
        loaded_model = pickle.load(open('finalized_model.sav', 'rb'))

        #scaler = pickle.load(open('scaler.sav', 'rb'))
        app.run()

