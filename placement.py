import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pickle
from flask import Flask,render_template,request,jsonify

data=pd.read_csv('placement.csv')

data1=data.drop(['sl_no','gender','ssc_b','hsc_b','hsc_s','degree_t','etest_p','specialisation','salary'],axis=1)
le=LabelEncoder()
data1['workex']=le.fit_transform(data1['workex'])
data1['status']=le.fit_transform(data1['status'])

x=data1.iloc[:,0:5].values
y=data1.iloc[:,5].values
train_x,test_x,train_y,test_y=train_test_split(x,y,test_size=0.3,random_state=0)
MLR=LinearRegression()
MLR.fit(train_x,train_y)

pickle.dump(MLR,open('placement.pkl','wb'))
model =pickle.load(open('placement.pkl','rb'))
app=Flask(__name__)
@app.route('/')
def home():
    return render_template('index1.html')

@app.route('/predict',methods=['POST'])

def predict():
    initial=[float(x) for x in request.form.values()]
    final=[np.array(initial)]
    prediction=model.predict(final)
    if(prediction>=0.5):
        return render_template('index1.html',prediction_text="Placed")
    else:
        return render_template('index1.html',prediction_text="Not Placed")

if __name__  == '__main__':
    app.run(debug=True)
