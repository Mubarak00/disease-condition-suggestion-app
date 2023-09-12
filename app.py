from flask import Flask, render_template, request
import pandas as pd
import joblib

app = Flask(__name__)

loaded_model = joblib.load('lr.pkl')


@app.route('/')
def index():
    return render_template('input.html')


@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get selected tests and their respective results
        selected_tests = request.form.getlist('selected_test')
        test_results = request.form.getlist('test_result')

        # Create a DataFrame with the input values
        input_data = pd.DataFrame({test: [float(result)] for test, result in zip(selected_tests, test_results)})
        prediction = loaded_model.predict(input_data)[0]

        # Determine the result based on the prediction
        if prediction == 'Diabetes':
            result = 'Your test results suggest a potential indication of diabetes'
        elif prediction == 'Pre-diabetes':
            result = 'Your test results suggest a potential indication of Pre-diabetes'
        else:
            result = 'Your test results suggest a potential indication of Non-Diabetes'

        return render_template('result.html', result=result)


if __name__ == '__main__':
    app.run(debug=True)
