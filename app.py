from flask import Flask, render_template, request, jsonify
from query_processor import QueryProcessor
from knowledge_base import KnowledgeBase
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize Flask app
app = Flask(__name__)

# Initialize services
knowledge_base = KnowledgeBase()
query_processor = QueryProcessor(knowledge_base=knowledge_base)
os.environ['HTTP_PROXY'] = 'http://127.0.0.1:10809'
os.environ['HTTPS_PROXY'] = 'http://127.0.0.1:10809'
# Routes
@app.route('/')
def index():
    documents = knowledge_base.get_all_documents()
    document_count = knowledge_base.get_document_count()
    return render_template('index.html', documents=documents, document_count=document_count)

@app.route('/api/query', methods=['POST'])
def api_query():
    data = request.get_json()
    query = data.get('query', '')
    result = query_processor.process_query(query)
    return jsonify(result)

@app.route('/api/documents', methods=['POST'])
def api_add_document():
    data = request.get_json()
    text = data.get('text', '')
    metadata = data.get('metadata', {})
    
    if not text:
        return jsonify({'success': False, 'error': 'Document text is required'})
    
    doc_id = knowledge_base.add_document(text, metadata)
    if doc_id:
        return jsonify({'success': True, 'document_id': doc_id})
    else:
        return jsonify({'success': False, 'error': 'Failed to add document'})

@app.route('/api/documents/<doc_id>', methods=['DELETE'])
def api_delete_document(doc_id):
    success = knowledge_base.delete_document(doc_id)
    if success:
        return jsonify({'success': True})
    else:
        return jsonify({'success': False, 'error': 'Document not found'})

@app.route('/api/documents', methods=['GET'])
def api_get_documents():
    documents = knowledge_base.get_all_documents()
    return jsonify({'success': True, 'documents': documents})

@app.route('/api/stats', methods=['GET'])
def api_get_stats():
    stats = query_processor.get_knowledge_base_stats()
    return jsonify({'success': True, 'stats': stats})

if __name__ == '__main__':
    host = os.getenv('FLASK_HOST', '0.0.0.0')
    port = int(os.getenv('FLASK_PORT', 5000))
    debug = os.getenv('FLASK_DEBUG', 'True').lower() == 'true'
    
    app.run(host=host, port=port, debug=debug)
