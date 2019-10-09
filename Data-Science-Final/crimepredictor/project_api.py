"""
Alex Tresselt
CS 7180
12/13/18
SPD Crime Prediction API
"""

import connexion
import numpy as np
from sklearn.externals import joblib
from sklearn.preprocessing import LabelEncoder

# Instantiate our Flask app object
app = connexion.FlaskApp(__name__, port=8080, specification_dir='swagger/')
application = app.app

# Classifier model
clf = joblib.load("./model/DTclassifier.joblib")

# Crime type label encoder
le = joblib.load("./model/le.joblib")

# Health Check (GET)
def health():
    try:
        predict(1,1,1,1)
    except:
        return {"Message" : "Service is unhealthy"}, 500

    return {"Message": "Service is OK"}


# Predict Top 5 Most Probably Crimes (POST)
def predict(beat, hour, day, month):
    # Predict top 5 most likely crimes
    proba = clf.predict_proba([[beat, hour, day, month]])
    top5 = np.argsort(proba, axis=1)[:,-5:]
    top5 = top5[0]

    # Transform results to labels
    results = []
    for t in top5:
        results.append(str(le.inverse_transform([t])))

    return {'predicted crimes' : results}

# Read the API definition for our service from the yaml file
app.add_api("project_api.yaml")

# Start the app
if __name__ == "__main__":
    app.run()
