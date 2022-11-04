import flask
import json
import pandas as pd
import pickle

model = pickle.load(open('model.pkl', 'rb'))

# create flask app
app = flask.Flask(__name__)

# create endpoint
@app.route('/predict', methods=['GET'])
def predict():
    # get data from request
    data = flask.request.get_json(force=True)
    data_df = pd.DataFrame(data, index=[0])
    vals=data_df.values
    # predict with model
    result = model.predict(vals)
    # return result
    return flask.jsonify(result.tolist())

# start flask app on port 8081
app.run(host='localhost', port=8081)




