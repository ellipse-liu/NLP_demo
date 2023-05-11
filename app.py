from flask import Flask, render_template, request
from model import sp_syllabler, onc_to_phon
import pickle

app = Flask(__name__)

piss_len = 45
shit_len = 19
piss_weights = 'piss_data/piss_weights.h5'
shit_weights = 'shit_data/shit_best_weights.h5'
piss_e2i = pickle.load(open('piss_data/e2i.pkl', 'rb'))
shit_e2i = pickle.load(open('shit_data/e2i.pkl', 'rb'))
shit_d2i = pickle.load(open('shit_data/d2i.pkl', 'rb'))

piss = sp_syllabler(piss_e2i, piss_len, 256, 256, len(piss_e2i) + 1)
shit = onc_to_phon(shit_e2i, shit_d2i, shit_len, 128, 500)

piss.model.load_weights(piss_weights)
shit.model.load_weights(shit_weights)

@app.route('/')
def home():
    return render_template('index.html', prediction=None)

@app.route('/predict_syl', methods=['POST'])
def predict_syl():
    input_data = request.form['input_data']
    prediction = piss.syllabify(input_data)
    return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
