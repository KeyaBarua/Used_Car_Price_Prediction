import pickle
from flask import Flask, request, jsonify, render_template
import numpy as np
import pandas as pd

application = Flask(__name__)
app = application

# Import Ridge Regressor
ridge_model = pickle.load(open('ridge_model.pkl', 'rb'))

# Creating Home Page
@app.route("/")
def index():
    return render_template('index.html')

@app.route("/predictdata", methods=['GET', 'POST'])
def predict_datapoint():
    print("Form Submitted")
    try:
        if request.method == "POST":
            make_year = int(request.form.get('make_year'))
            mileage_kmpl = float(request.form.get('mileage_kmpl'))
            engine_cc = int(request.form.get('engine_cc'))
            fuel_type = request.form.get('fuel_type')
            owner_count = int(request.form.get('owner_count'))
            brand = request.form.get('brand')
            transmission = float(request.form.get('transmission'))
            service_history = float(request.form.get('service_history'))
            accidents_reported = int(request.form.get('accidents_reported'))
            insurance_valid = float(request.form.get('insurance_valid'))

            # DataFrame for prediction
            input_df = pd.DataFrame([{
                "make_year": make_year,
                "mileage_kmpl": mileage_kmpl,
                "engine_cc": engine_cc,
                "fuel_type": fuel_type,
                "owner_count": owner_count,
                "brand": brand,
                "transmission": transmission,
                "service_history": service_history,
                "accidents_reported": accidents_reported,
                "insurance_valid": insurance_valid
            }])


            result = ridge_model.predict(input_df)
            prediction = round(result[0], 2)

            return render_template('home.html', result=prediction)
        
    except Exception as e:
        return render_template('home.html', result=f"Error: {e}")
    
    else:
        return render_template('home.html')

if __name__=="__main__":
    app.run(host="0.0.0.0")