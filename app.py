from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

@app.route('/api/chat', methods=['POST'])
def chat():
    data = request.get_json()
    user_input = data.get("message", "").lower()

    if user_input == "hi":
        return jsonify({"response": "Hello Kaela Ganda! "})
    else:
        return jsonify({"response": "Sorry, I don't understand."})

if __name__ == '__main__':
    app.run(debug=True)