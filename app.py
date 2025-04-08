from flask import Flask, render_template, request
import joblib

# Load the trained model and vectorizer
model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        news = request.form['news']
        vect_text = vectorizer.transform([news])
        prediction = model.predict(vect_text)

        result = '✅ REAL News' if prediction[0] == 'REAL' else '🚨 FAKE News'
        return render_template('index.html', prediction=result)

if __name__ == '__main__':
    app.run(debug=True)
