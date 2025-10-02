import streamlit as st
import requests

st.title("📄 PDF Q&A Bot (FAISS + Groq)")
st.title("Created by AI Engineer  Syed Rahman")

backend_url = "https://pdf-backend.onrender.com"


# Step 1: Upload PDF
uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

if uploaded_file:
    try:
        response = requests.post(
            f"{backend_url}/upload_pdf/",
            files={"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")}
        )
        if response.status_code == 200:
            st.success(response.json()["message"])
        else:
            st.error(f"❌ Error uploading PDF: {response.text}")
    except Exception as e:
        st.error(f"❌ Exception during upload: {e}")

# Step 2: Ask question
question = st.text_input("Ask a question about the PDF:")

if st.button("Get Answer"):
    if not uploaded_file:
        st.warning("⚠️ Please upload a PDF first!")
    elif not question.strip():
        st.warning("⚠️ Please type a question!")
    else:
        try:
            response = requests.post(
                f"{backend_url}/ask/",
                data={"question": question}
            )
            if response.status_code == 200:
                st.success("✅ Answer received!")
                st.write("🤖 Answer:", response.json()["answer"])
            else:
                st.error(f"❌ Error: {response.text}")
        except Exception as e:
            st.error(f"❌ Exception while asking question: {e}")
