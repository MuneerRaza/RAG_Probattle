import os
import faiss
import json
import pickle
from langchain.chat_models import init_chat_model
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_unstructured import UnstructuredLoader
from langchain_community.retrievers import BM25Retriever
from langchain.schema import Document
from langchain import hub
from fastapi.responses import StreamingResponse
from langchain_community.document_loaders import PyPDFLoader
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List


if not os.environ.get("GROQ_API_KEY"):
  os.environ["GROQ_API_KEY"] = ""

# Initialize models
llm = init_chat_model("llama3-8b-8192", model_provider="groq")
embeddings = HuggingFaceEmbeddings(model_name="nomic-ai/nomic-embed-text-v1.5", model_kwargs={"trust_remote_code": True})

# Paths
path = r"C:\Users\Muneer\Downloads\pa-2024-25.pdf"
json_path = r"C:\Users\Muneer\Downloads\courses_info.json"
vectorstore_path = "new_vectorstore"

def convert_json_to_documents(json_path):
    """Loads JSON and converts each object into a LangChain Document."""
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    docs = []
    for obj in data:
        text = (
            f"Class Name: {obj['name']}\n"
            f"Faculty: {obj['faculty']}\n"
            f"Time: {obj['start_time']}\n"
            f"Days: {obj['days']}\n"
            f"Enrolled Students: {obj['std_enrolled']}\n"
            f"Class Limit: {obj['class_limit']}\n"
            f"Class Code: {obj['class_code']}\n"
        )
        docs.append(Document(page_content=text, metadata={"source": "json"}))
    
    return docs

def load_and_store_embeddings():
    """Loads a PDF, splits text, and stores embeddings in FAISS."""
    # loader_local = UnstructuredLoader(
    #    file_path=path,
    #    strategy="hi_res",
    # )
    # docs = []
    # for doc in loader_local.lazy_load():
    #     docs.append(doc)
    loader = PyPDFLoader(path)
    docs = []
    for page in loader.lazy_load():
        docs.append(page)

    json_docs = convert_json_to_documents(json_path)

    docs.extend(json_docs)

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    all_splits = text_splitter.split_documents(docs)

    vector_store = FAISS.from_documents(all_splits, embeddings)
    vector_store.save_local(vectorstore_path)

    with open(f"{vectorstore_path}/metadata.pkl", "wb") as f:
        pickle.dump(all_splits, f)
    
    print("Embeddings saved successfully!")

# Run this only once to generate embeddings
if not os.path.exists(vectorstore_path):
    load_and_store_embeddings()
else:
    print("Embeddings already exist. Skipping storage.")


def load_vector_store():
    """Loads the FAISS vector store and associated metadata."""
    vector_store = FAISS.load_local(vectorstore_path, embeddings, allow_dangerous_deserialization=True)

    with open(f"{vectorstore_path}/metadata.pkl", "rb") as f:
        metadata = pickle.load(f)

    return vector_store, metadata

vector_store, all_splits = load_vector_store()



# Use retrieval and generation as before
prompt = hub.pull("rlm/rag-prompt")

def re_rank(retrieved_docs, question):
    scores = []
    for doc in retrieved_docs:
        messages = [
            {"role": "system", "content": "Score this document's relevance to the question from 0-10.0"},
            {"role": "user", "content": f"Question: {question}\nDocument: {doc.page_content}"}
        ]
        response = llm.invoke(messages).content
        try:
            score = float(''.join(filter(str.isdigit, response)))  # Extract numeric score
            scores.append((score, doc))
        except ValueError:
            scores.append((0, doc))

    # Fix: Sort based on score
    scores.sort(key=lambda x: x[0], reverse=True)
    retrieved_docs = [doc for _, doc in scores]
    
    return retrieved_docs


from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class Query(BaseModel):
    question: str

class ResponseModel(BaseModel):
    answer: str
    sources: List[str] = []

from fastapi.responses import JSONResponse

@app.post("/generate")
async def generate(query: Query):
    question = query.question
    question += "\n Try to answer from the context according to question but If the question above is very irrelevant to provided context, then don't answer it."
    
    retrieved_docs = vector_store.similarity_search(question, top_k=5)
    retriever = BM25Retriever.from_documents(all_splits)
    bm_docs = retriever.invoke(question)
    retrieved_docs.extend(bm_docs)

    retrieved_docs = re_rank(retrieved_docs, question)
    docs_content = "\n\n".join(doc.page_content for doc in retrieved_docs)

    # Collect source metadata
    sources = [
        {"source": doc.metadata.get("source", "Unknown"), "page": doc.metadata.get("page", "N/A"), "page_content": doc.page_content}
        for doc in retrieved_docs
    ]

    messages = prompt.invoke({"question": question, "context": docs_content})

    async def generate_stream():
        for chunk in llm.stream(messages):
            yield chunk.content

    response = StreamingResponse(generate_stream(), media_type="text/plain")
    response.headers["X-Sources"] = json.dumps(sources)  # Send sources in headers
    return response
