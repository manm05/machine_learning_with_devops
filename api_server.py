from flask import Flask, request, redirect, url_for, flash, jsonify
import numpy as np
import pickle as p
import json
from sklearn.externals import joblib


app = Flask(__name__)


@app.route('/api/', methods=['POST'])
def predict():
    data = request.get_json()
    prediction = np.array2string(model.predict(data))

    return jsonify(prediction)

if __name__ == '__main__':
    modelfile = 'customer_churn_pipeline_test.pkl'
    model = joblib.load(open(modelfile))
    app.run(debug=True, host='0.0.0.0', port=7788)