from flask import Flask, request, jsonify
from g4l.local import LocalEngines
import json

app = Flask(__name__)

# Read the engine parameters and create the engine
engine = LocalEngines.create_engine(
    model="mistral-7b-instruct-v0.2.Q5_K_M",
    n_gpu_layers = -1,  # use all GPU layers
    cores      = 8    # use all CPU cores
)

# Define a route for the API
@app.route('/api/chat', methods=['POST'])
def chat():
    try:
        messages = request.json['messages']
        stop = request.json.get('stop', None)
        temperature = request.json.get('temperature', 0.0)

        # Create the completion and get the response
        response = engine.create_chat_completion(
            messages=messages, 
            stream=False,
            stop=stop,
            temperature=temperature
        )

        # Access the 'choices' key as a dictionary, not as an attribute
        first_choice_content = response['choices'][0]['message']['content']
        print(f"Response: {first_choice_content}")

        # Return the response as a JSON object
        return jsonify({'choices': [{'message': {'content': first_choice_content}}]}), 200
    except Exception as e:
        print(f"Error: {e}")
        error_info = {
            'error': str(e)
        }
        return jsonify(error_info), 500

# Run the web server
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
