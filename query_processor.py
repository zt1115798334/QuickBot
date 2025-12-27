from knowledge_base import KnowledgeBase

class QueryProcessor:
    def __init__(self, knowledge_base=None):
        if knowledge_base:
            self.knowledge_base = knowledge_base
        else:
            self.knowledge_base = KnowledgeBase()
    
    def process_query(self, query, top_k=3):
        """Process a user query and generate an answer"""
        if not query or not isinstance(query, str):
            return {
                'success': False,
                'error': 'Invalid query format'
            }
        
        # Search for relevant documents
        search_results = self.knowledge_base.search(query, top_k=top_k)
        
        if not search_results:
            return {
                'success': True,
                'answer': 'I don\'t have information about this topic in my knowledge base.',
                'sources': []
            }
        
        # Generate answer from the most relevant document
        # For now, we'll just return the most relevant document text
        most_relevant = search_results[0]
        answer = most_relevant['text']
        
        # Prepare sources information
        sources = []
        for i, result in enumerate(search_results):
            sources.append({
                'id': result['id'],
                'text': result['text'],
                'similarity': result['similarity'],
                'metadata': result.get('metadata', {})
            })
        
        return {
            'success': True,
            'answer': answer,
            'sources': sources
        }
    
    def get_knowledge_base_stats(self):
        """Get statistics about the knowledge base"""
        return {
            'document_count': self.knowledge_base.get_document_count()
        }
