from flask import Flask, request, jsonify
from main import RAGAgent
import os

app = Flask(__name__)

# Production configuration
app.config['DEBUG'] = False
app.config['TESTING'] = False

# Initialize RAG agent
rag_agent = RAGAgent()

@app.route('/chat', methods=['POST'])
def chat():
    data = request.get_json()
    if not data or 'question' not in data:
        return jsonify({'error': 'Missing question in request'}), 400

    question = data['question']
    chat_history = data.get('chat_history', [])

    try:
        response = rag_agent.run(question, chat_history)
        return jsonify(response)
    except Exception as e:
        app.logger.error(f'Error processing question: {str(e)}')
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'healthy'}), 200

if __name__ == '__main__':
    # For development
    app.run(debug=True)
else:
    # For production (when using WSGI)
    application = app