import logging
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage

from RAG_Engine.config import (
    GROQ_API_KEY,
    LLM_MODEL,
    LLM_TEMPERATURE,
    LLM_MAX_TOKENS,
    WEAK_EVIDENCE_MSG,
)
from RAG_Engine.citations import build_evidence_context, format_citations_block

logger = logging.getLogger(__name__)

# System prompts 

RESOLVE_SYSTEM_PROMPT = """\
أنت مساعد قانوني مصري. لديك سياق محادثة قانونية سابقة بين المستخدم والمساعد.
السؤال الحالي قصير أو غامض ويعتمد على ما سبق.
مهمتك: أعد صياغة السؤال الحالي كسؤال قانوني مستقل ومكتمل يمكن فهمه دون الحاجة للسياق.
- احتفظ بالمعنى الأصلي تماماً
- أضف التفاصيل الضرورية من السياق لجعل السؤال واضحاً ومكتفياً بذاته
- أعد السؤال المُعاد صياغته فقط، بدون أي شرح أو مقدمة
- لا تتجاوز جملتين
"""

REWRITE_SYSTEM_PROMPT = """\
أنت خبير قانوني مصري متخصص في صياغة الاستفسارات القانونية.
مهمتك: إعادة صياغة السؤال القانوني المدخل بشكل أوسع وأدق يشمل المصطلحات القانونية الرسمية التي قد ترد في النصوص القانونية.
- أبقِ المعنى الأصلي كما هو
- أضف مصطلحات قانونية ذات صلة
- استخدم صياغة تساعد في البحث النصي  
- أعد السؤال المعاد صياغته فقط، بدون أي شرح أو مقدمة
- لا تتجاوز 2-3 جملة قصيرة
"""

ANSWER_SYSTEM_PROMPT = """\
أنت مستشار قانوني متخصص في القانون المصري. مهمتك تقديم إجابات قانونية دقيقة وموثوقة.

قواعد صارمة يجب الالتزام بها:
1. أجب باللغة العربية الفصحى الرسمية دائماً.
2. استند فقط إلى النصوص القانونية المُقدمة في قسم [الأدلة القانونية]. لا تختلق أرقام مواد أو أحكام.
3. اذكر رقم المادة والقانون بوضوح عند استشهادك بنص قانوني.
4. إذا كانت الأدلة غير كافية أو غير ذات صلة، أجب بالعبارة التالية فقط دون إضافة أي شيء آخر:
   "لم يتم العثور على نص قانوني صريح ضمن البيانات المتاحة."
5. لا تُقدم آراء شخصية أو توصيات بالتقاضي.
6. هيكل الإجابة:
   - الحكم القانوني المختصر
   - النص القانوني المستشهد به
   - المصدر (القانون + رقم المادة إن وجد)

[الأدلة القانونية]
{evidence_context}
"""


class LegalGenerator:
    """
    Handles query rewriting and grounded answer generation via ChatGroq.
    """

    def __init__(self) -> None:
        logger.info(f"Initializing ChatGroq — model={LLM_MODEL}")
        self.llm = ChatGroq(
            api_key=GROQ_API_KEY,
            model=LLM_MODEL,
            temperature=LLM_TEMPERATURE,
            max_tokens=LLM_MAX_TOKENS,
        )

    # Follow-up Resolution

    def resolve_followup(self, question: str, context: str) -> str:
        """
        Expand an ambiguous Arabic follow-up into a standalone legal question.

        Called only when ConversationMemory.is_followup() returns True.
        Passes the recent conversation context so the LLM can infer the subject.

        Examples:
            context: "المستخدم: ما شروط حضانة الأم؟ / المساعد: ..."
            question: "ولو رفض؟"
            → "ولو رفض الأب تسليم الأطفال للأم الحاضنة، ما الإجراءات القانونية؟"

        Args:
            question: The short, ambiguous follow-up question.
            context:  Formatted conversation history from ConversationMemory.

        Returns:
            Resolved standalone question. Falls back to original on any error.
        """
        try:
            messages = [
                SystemMessage(content=RESOLVE_SYSTEM_PROMPT),
                HumanMessage(
                    content=(
                        f"[سياق المحادثة السابقة]\n{context}\n\n"
                        f"[السؤال الحالي]\n{question}"
                    )
                ),
            ]
            response = self.llm.invoke(messages)
            resolved = response.content.strip()
            logger.debug(
                f"Follow-up resolved: '{question[:40]}' → '{resolved[:70]}'"
            )
            return resolved
        except Exception as e:
            logger.warning(f"Follow-up resolution failed ({e}), using original question")
            return question

    # Query Rewriting 

    def rewrite_query(self, question: str) -> str:
        """
        Expand a short Arabic legal question with richer legal terminology.
        This bridges the gap between colloquial user queries and formal legal text.

        Args:
            question: Original Arabic legal question from the user.

        Returns:
            Rewritten/expanded query string. Falls back to original on error.
        """
        try:
            messages = [
                SystemMessage(content=REWRITE_SYSTEM_PROMPT),
                HumanMessage(content=f"السؤال الأصلي: {question}"),
            ]
            response  = self.llm.invoke(messages)
            rewritten = response.content.strip()
            logger.debug(f"Query rewritten: '{question[:40]}' → '{rewritten[:60]}'")
            return rewritten
        except Exception as e:
            logger.warning(f"Query rewrite failed ({e}), using original query")
            return question

    #  Answer Generation 
    def generate_answer(self, question: str, chunks: list[dict]) -> dict:
        """
        Generate a grounded Arabic legal answer from the top reranked chunks.

        Args:
            question: The user's original Arabic question.
            chunks:   Top-N reranked chunk dicts (each has text + metadata).

        Returns:
            {
                "answer"    : str   — full Arabic response
                "citations" : str   — formatted citation block
                "is_grounded: bool  — False if fallback message was returned
            }
        """
        if not chunks:
            return {
                "answer":     WEAK_EVIDENCE_MSG,
                "citations":  "",
                "is_grounded": False,
            }

        # Build evidence context from chunk texts + metadata
        evidence_context = build_evidence_context(chunks)

        # Inject evidence into system prompt
        system_content = ANSWER_SYSTEM_PROMPT.format(
            evidence_context=evidence_context
        )

        try:
            messages = [
                SystemMessage(content=system_content),
                HumanMessage(content=f"السؤال: {question}"),
            ]
            response = self.llm.invoke(messages)
            answer   = response.content.strip()

            # Detect if LLM returned the fallback phrase
            is_grounded = WEAK_EVIDENCE_MSG not in answer

            # Build citation block for grounded answers
            citations = format_citations_block(chunks) if is_grounded else ""

            logger.debug(
                f"Answer generated — grounded={is_grounded}, "
                f"length={len(answer)} chars"
            )
            return {
                "answer":      answer,
                "citations":   citations,
                "is_grounded": is_grounded,
            }

        except Exception as e:
            logger.error(f"Answer generation failed: {e}")
            return {
                "answer":      WEAK_EVIDENCE_MSG,
                "citations":   "",
                "is_grounded": False,
            }
