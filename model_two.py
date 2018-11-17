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
#create the model
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.layers import Dropout
from tensorflow.python.keras.optimizers import Adam #will do the back prop to optimize the model

#Generalized Pattern for the Sequential Model is in steps as follows
    
    ##Create Model##
model= Sequential() #simple Sequential Model(layers are added one after the other)
    
    ##Add Layers##    
model.add(Dense(304,activation="relu",kernel_initializer='uniform',input_dim=304))
        #'1'- 0ne neuron/
        #'input_shape=(2,)'means that we are inputting (x,y) data pairs
        #'activation="sigmoid"' is the activation function of the neuron
        #model.add(Dense(10,activation="tanh"))
#model.add(Dense(12,kernel_initializer='normal',activation="relu"))

model.add(Dense(12,kernel_initializer='normal',activation="relu"))
#lets add a Dropout layer to reduce the overfitting of model..change the para value to change the accuracy
#model.add(Dropout(0.5))
model.add(Dense(12,kernel_initializer='normal',activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(12,kernel_initializer='normal',activation="relu"))
model.add(Dense(12,kernel_initializer='normal',activation="relu"))

model.add(Dense(1,kernel_initializer='normal',activation="sigmoid"))
    
    ##Compile Model##
model.compile(Adam(lr=0.01),'binary_crossentropy',metrics=['accuracy'])
        #'Adam(lr=0.05)' is used to minimize the error
        #'binary_crossentropy' function is used to calculate the loss
        #'metrics=['accuracy']' is the metric we want to optimize
    
   
#################################################################################################################
#training and evalutaing the model
model.fit(X_train,Y_train,epochs=250,verbose=0)
        #'epochs=100' how many times your are running through the data
    
        ##Evaluate Performance##
eval_result=model.evaluate(X_test,Y_test)
    
print("\n\nTest Loss:",eval_result[0],"Test Accuracy :",eval_result[1])

import h5py
model_json=model.to_json()
with open("model.json","w") as json_file :
    json_file.write(model_json)

model.save_weights("model.h5")     

#################################################################################################################

################################################################################################################
@app.route('/detect/', methods=["POST"])
def check_for_dementia():    
   json_ = request.json 
   print(json_,"\n\n\n")  
   dataset = pd.DataFrame(json_, index=[0]) 
   print("Dataset : \n\n",dataset)
     
   #score = model.predict(dataset)
   #print("\n",score,"\n")
   score = 0.09
   isDaignosed = (score > 0.05)
   data = {
        'isDaignosed'  : np.bool(isDaignosed),
        'scorre' : np.float64(score)
   }

   js = json.dumps(data)

   response = Response(js, status=200, mimetype='application/json')
   return response
################################################################################################################
if __name__ == "__main__":
        #model_columns = joblib.load('model_columns.pkl')
        model = load_trained_model("model.h5")
        #scaler = pickle.load(open('scaler.sav', 'rb'))
        app.run()

