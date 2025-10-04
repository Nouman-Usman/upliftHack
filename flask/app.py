from flask import Flask, request, jsonify
from main import RAGAgent
import os

app = Flask(__name__)

# Production configuration
app.config['DEBUG'] = False
app.config['TESTING'] = False

# Lazy initialize RAG agent to avoid heavy startup failures
rag_agent = None


def get_rag_agent():
    global rag_agent
    if rag_agent is None:
        rag_agent = RAGAgent()
    return rag_agent

@app.route('/chat', methods=['POST'])
def chat():
    data = request.get_json()
    if not data or 'question' not in data:
        return jsonify({'error': 'Missing question in request'}), 400

    question = data['question']
    chat_history = data.get('chat_history', [])

    try:
        agent = get_rag_agent()
        response = agent.run(question, chat_history)
        return jsonify(response)
    except Exception as e:
        app.logger.error(f'Error processing question: {str(e)}')
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'healthy'}), 200

@app.route('/', methods=['GET'])
def root():
    return jsonify({'message': 'Flask app is running'}), 200

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    host = os.environ.get('HOST', '0.0.0.0')
    app.run(host=host, port=port, debug=False)