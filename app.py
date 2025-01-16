from flask import Flask, request, url_for, redirect, render_template,jsonify
import numpy as np
import pickle

from flask_sqlalchemy import SQLAlchemy

model = pickle.load(open('model.pkl', 'rb'))

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = ""


@app.route('/')
def hello_world():
    return render_template('index.html')


@app.route('/products')
def products():
    return 'products'


@app.route('/predict', methods=['POST'])
def predict():
    age = request.form.get('age', None)
    diabetic = request.form.get('diabetic', None)
    highbloodpressure = request.form.get('highbloodpressure', None)
    sex = request.form.get('sex', None)
    smoking=request.form.get('smoking', None)

    if age is None or diabetic is None or highbloodpressure is None or sex is None:
        return 'Error: Missing form data', 400
    
    if (diabetic == "yes"):
        diabetic = 1
    else:
        diabetic = 0
    if (highbloodpressure == "yes"):
        highbloodpressure = 1
    else:
        highbloodpressure = 0
    if (sex == "male"):
        sex = 1
    else:
        sex = 0
    if(smoking == "yes"):
        smoking = 1
    else:
        smoking = 0
    age = float(age)
    diabetic = float(diabetic)
    highbloodpressure = float(highbloodpressure)
    sex = float(sex)
    smoking = float(smoking)


    result1 = model.predict([[age, diabetic, highbloodpressure, sex,smoking]])[0]
    result2=round(result1,2)*100
    result3=round(result2,2)
    result=f"{result3} % chance of a heart stroke"
    return render_template('index.html', **locals())
    

if __name__ == "__main__":
    app.run(debug=True, port=8000)