import os
from PyPDF2 import PdfReader
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.llms.base import LLM
from langchain_community.embeddings import HuggingFaceEmbeddings
from groq import Groq


# Load .env
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError("Please set your GROQ_API_KEY in .env")


# Initialize Groq client
client = Groq(api_key=GROQ_API_KEY)


# Custom LLM wrapper for Groq
class GroqLLM(LLM):
    @property
    def _llm_type(self) -> str:
        return "groq"

    def _call(self, prompt: str, stop=None) -> str:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
        )
        return response.choices[0].message.content


# Instantiate LLM
llm = GroqLLM()

# Initialize HuggingFace embeddings for FAISS
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


def process_pdf(file) -> FAISS:
    """
    Extract text from PDF, split into chunks, and store in FAISS vectorstore.
    """
    pdf_reader = PdfReader(file)
    text = ""
    for page in pdf_reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text

    # Split text into chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_text(text)

    # Create FAISS vectorstore with embeddings
    vectorstore = FAISS.from_texts(chunks, embedding=embeddings)
    print(f"✅ FAISS vectorstore created with {len(chunks)} chunks.")
    return vectorstore


def get_answer(query: str, vector_store) -> str:
    """
    Retrieve relevant chunks from FAISS and generate answer using Groq LLM
    """
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})
    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
    try:
        answer = qa_chain.run(query)
        return answer
    except Exception as e:
        return f"❌ Error generating answer: {str(e)}"
