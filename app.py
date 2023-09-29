from flask import Flask, render_template, request
import joblib

# Initialize the Flask app
app = Flask(__name__)

# Load your trained model from the pickle file
model = joblib.load('regression_model.joblib')

# Define a route for the home page
@app.route('/')
def home():
    return render_template('index.html')  # Create an HTML form for user input

# Define a route to handle the prediction
@app.route('/predict', methods=['POST'])
def predict():
    # Get user input from the form
    input_features = [
        float(request.form['MedInc']),
        float(request.form['HouseAge']),
        float(request.form['AveRooms']),
        float(request.form['AveBedrms']),
        float(request.form['Population']),
        float(request.form['AveOccup']),
        float(request.form['Latitude']),
        float(request.form['Longitude'])
    ]

    # Make a prediction using the loaded model
    predicted_value = model.predict([input_features])[0]

    # Render the result on an HTML page
    return render_template('result.html', prediction=predicted_value)

if __name__ == '__main__':
    app.run(debug=True)
