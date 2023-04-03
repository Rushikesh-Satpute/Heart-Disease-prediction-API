from flask import Flask,request,jsonify
import pickle
import numpy as np

model = pickle.load(open('model.pkl','rb'))

app = Flask(__name__)

@app.route('/')
def home():
    return "Heart Disease Prediction"

@app.route('/predict',methods=['POST'])
def predict():
    age = request.form.get('age')
    sex = request.form.get('sex')
    chest_pain = request.form.get('chest_pain')
    resting_blood_pressure = request.form.get('resting_blood_pressure')
    heartbeat_rate = request.form.get('heartbeat_rate')
    fasting_blood_sugar = request.form.get('fasting_blood_sugar')
    maximum_heart_rate = request.form.get('maximum_heart_rate')
    exercise_induced_angina = request.form.get('exercise_induced_angina')

    input_query = np.array([[age,sex,chest_pain,resting_blood_pressure,heartbeat_rate,fasting_blood_sugar,maximum_heart_rate,exercise_induced_angina]])
    result = model.predict(input_query)[0]

    #if result == 0:
        #return jsonify({'Your Heart is Healthy': str(result)})
    return jsonify({'prediction':str(result)})

if __name__ == '__main__':
    app.run(debug=True)