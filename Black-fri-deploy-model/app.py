import numpy as np
import joblib 
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb

#---------------------------importing data and training the ocuntvectorizer----------------------------------------------
# df=pd.read_csv('preprocessed_twitter_data.csv')

# df = df.dropna(subset=['clean_tweet'])
# df = df.reset_index(drop=True)

# bow_vectorizer = CountVectorizer(max_df=0.90,min_df=2,max_features=5000, stop_words='english')

# bow_vectorizer.fit(df['clean_tweet'])
#----------------------------------Defining preprocessing functions-------------------------------------------
# gender_dict = {'F':0, 'M':1}
# Gender = Gender.apply(lambda x: gender_dict[x])


# cols = ['Age', 'City_Category', 'Stay_In_Current_City_Years']
# le = LabelEncoder()
# for col in cols:
#   col = le.fit_transform(col)


model=joblib.load('Black_friday_model.joblib')
#-----------------------------------------------------Flask App---------------------------------------------------------------------------------------


from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

@app.route('/')
def hello_world():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def process_text():
    gender = str(request.form.get("Gender"))
    age = int(request.form.get("Age"))
    occ = int(request.form.get("Occupation"))
    city_cat = str(request.form.get("City_Category"))
    siccy = int(request.form.get("Stay_In_Current_City_Years"))
    mstatus = int(request.form.get("Marital_Status"))
    pro1 = int(request.form.get("Product_Category_1"))
    pro2 = float(request.form.get("Product_Category_2"))
    pro3 = float(request.form.get("Product_Category_3"))

    if gender == 'M':
        gender = 1
        gender = int(gender)
    else:
        gender = 0
        gender = int(gender)

    city_cat=int(city_cat)
    cols = [[age, int(city_cat), siccy]]
    le = LabelEncoder()
    for col in cols:
       col = le.fit_transform(col)
    li = [gender,cols[0][0],occ,cols[0][1],cols[0][2],mstatus,pro1,pro2,pro3]
   
    pred = model.predict([li])
    finalpred = int(pred[0])
    
    # return jsonify({'prediction':pred})
    return render_template('index.html',pred="The Purchase is : "+str(finalpred))

if __name__ == '__main__':
    app.run(debug=True)

