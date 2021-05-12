from flask import Flask, request, render_template
import pickle
import numpy as np

# create an app
app = Flask(__name__)
app.config['DEBUG']=True
# load the model
model = pickle.load(open('houseprice_model.pkl', 'rb'))


@app.route('/')
def home():
   return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
   float_features = [float(x) for x in request.form.values()]
   final_features = [np.array(float_features)]
   prediction = model.predict(final_features)
   output = round(prediction[0], 2)
   return render_template('index.html',
                          prediction_text='The price of the house should be $ {}'.format(output))

if __name__ == '__main__':
   app.run(debug=True)
