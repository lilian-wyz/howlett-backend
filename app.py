from flask import Flask, request, jsonify
from models import get_prediction

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    date = data.get('date')
    model_name = data.get('model')

    if not date or not model_name:
        return jsonify({'error': 'Missing required fields (date or model).'}), 400

    try:
        # Get prediction from the selected model
        prediction = get_prediction(date, model_name)
        return jsonify({'prediction': prediction})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
