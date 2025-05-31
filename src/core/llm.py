"""LLM configuration and setup."""
import logging
from langchain_ollama.chat_models import ChatOllama
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os

logger = logging.getLogger(__name__)
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")

class LLMManager:
    """Manages LLM configuration and prompts."""
    
    def __init__(self, model_name: str = "meta-llama/llama-4-scout-17b-16e-instruct"):
        self.model_name = model_name
        # self.llm = ChatOllama(model=model_name)
        self.llm = ChatGroq(
                    model_name=model_name,
                    temperature=0.7,
                    api_key=groq_api_key)
        
    def get_query_prompt(self) -> PromptTemplate:
        """Get query generation prompt."""
        return PromptTemplate(
            input_variables=["question"],
            template="""You are an AI language model assistant. Your task is to generate 2
            different versions of the given user question to retrieve relevant documents from
            a vector database. By generating multiple perspectives on the user question, your
            goal is to help the user overcome some of the limitations of the distance-based
            similarity search. Provide these alternative questions separated by newlines.
            Original question: {question}"""
        )
    
    def get_rag_prompt(self) -> ChatPromptTemplate:
        """Get RAG prompt template."""
        template = """Answer the question based ONLY on the following context:
        {context}
        Question: {question}
        """
        return ChatPromptTemplate.from_template(template) 