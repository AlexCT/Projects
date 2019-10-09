"""
Alex Tresselt
CS 7180
12/13/18
API for Response Time Predictor
"""

import connexion
import numpy as np
from sklearn.externals import joblib
from sklearn.preprocessing import LabelEncoder

# Instantiate our Flask app object
app = connexion.FlaskApp(__name__, port=8080, specification_dir='swagger/')
application = app.app

# Classifier model
clf = joblib.load("./model/RTPmodel.joblib")

# Crime type label encoder
le = joblib.load("./model/RTPle.joblib")

# Health Check (GET)
def health():
    try:
        predict(1,1,1)
    except:
        return {"Message" : "Service is unhealthy"}, 500

    return {"Message": "Service is OK"}


# Predict Response Time given the priority, calltype, and precinct.
def predict(priority, calltype, precinct):
    result = clf.predict([[priority, calltype, precinct]])

    return {'predicted response time (minutes)' : result[0]}

# Read the API yaml file
app.add_api("rtp_api.yaml")

# Start the app
if __name__ == "__main__":
    app.run()
