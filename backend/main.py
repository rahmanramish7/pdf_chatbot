from fastapi import FastAPI, UploadFile, Form
from fastapi.responses import JSONResponse
from io import BytesIO
from qa_engine import process_pdf, get_answer
from dotenv import load_dotenv
import os
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
app = FastAPI()

# Global vectorstore
vector_store = None

@app.post("/upload_pdf/")
async def upload_pdf(file: UploadFile):
    global vector_store
    try:
        pdf_bytes = await file.read()
        vector_store = process_pdf(BytesIO(pdf_bytes))  # fix: use BytesIO
        return {"message": "✅ PDF processed and indexed with FAISS!"}
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)


@app.post("/ask/")
async def ask_question(question: str = Form(...)):
    global vector_store
    try:
        if vector_store is None:
            return {"answer": "❌ No PDF uploaded yet."}
        answer = get_answer(question, vector_store)
        return {"answer": answer}
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
