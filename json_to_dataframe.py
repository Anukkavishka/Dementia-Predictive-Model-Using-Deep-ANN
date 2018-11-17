import requests
import numpy as np
import pandas as pd
from flask import Flask, json, jsonify,request,Response

app = Flask(__name__)

#r = requests.get('http://0.0.0.0/detect')


#j = r.json()

#df = pd.DataFrame([[d['v'] for d in x['c']] for x in j['rows']],columns=[d['label'] for d in j['cols']])

################################################################################################################
@app.route('/detect/', methods=["POST"])
def check_for_frauds():    
   json_ = request.json
   print(json_,'\n\n')   
   dataset = pd.DataFrame(json_, index=[0]) 
   print(dataset.head())
   print("\nServer Request Came Through\n") 
     
   #score = model.predict(test_scaled_rec)
   #isFraud = (score > 0.05)
   data = {
        'isFraud'  : np.bool(0.09),
        'scorre' : np.float64(123)
   }

   js = json.dumps(data)
   #dataset_conv=pd.DataFrame(dataset.to_json(orient='records'),index=[0])

   
   print("\nafter converting the json into a dataframe :\n\n")
   print(dataset.to_json())


   response = Response(js, status=200, mimetype='application/json')
   return response
 
###############################################################################################################
if __name__ == "__main__":
        #model_columns = joblib.load('model_columns.pkl')
        #model = load_trained_model("model.h5")
        #scaler = pickle.load(open('scaler.sav', 'rb'))
        app.run()                  
               