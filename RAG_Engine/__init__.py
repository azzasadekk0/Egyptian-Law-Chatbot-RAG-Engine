"""
__init__.py — Public API for the RAG_Engine package.

Stateless usage:
    from RAG_Engine import EgyptianLegalRAG

    rag = EgyptianLegalRAG()
    result = rag.query("ما عقوبة السرقة في القانون المصري؟")
    print(result.answer)

Conversational usage (with memory):
    result1 = rag.query("ما شروط الحضانة بعد الطلاق؟", session_id="user_1")
    result2 = rag.query("ولو رفض الأب؟",               session_id="user_1")
    # Follow-up is automatically resolved using conversation history
"""

from RAG_Engine.pipeline import EgyptianLegalRAG, RAGResult
from RAG_Engine.memory   import ConversationMemory

__all__ = ["EgyptianLegalRAG", "RAGResult", "ConversationMemory"]
__version__ = "1.1.0"
