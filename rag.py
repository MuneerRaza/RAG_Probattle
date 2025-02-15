import asyncio
from langchain_ollama import ChatOllama
from langchain_huggingface import HuggingFaceEmbeddings
# from langchain_qdrant import QdrantVectorStore
# from qdrant_client import QdrantClient
from langchain import hub
from langchain_unstructured import UnstructuredLoader
from langchain_community.document_loaders import CSVLoader, UnstructuredExcelLoader, UnstructuredWordDocumentLoader, UnstructuredPowerPointLoader, TextLoader, UnstructuredMarkdownLoader, JSONLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import START, StateGraph
from typing_extensions import List, TypedDict
from langchain_core.vectorstores import InMemoryVectorStore


llm = ChatOllama(
    model="llama3.2",
)

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
vector_store = InMemoryVectorStore(embeddings)
# client = QdrantClient(":memory:")
# client.recreate_collection("rag_docs")
# qdrant_vector_store = QdrantVectorStore(
#     client=client,
#     collection_name="rag_docs",
#     embedding=embeddings,
# )


def load_pdf(path: str):
    loader = UnstructuredLoader(file_path=path,strategy="hi_res")
    docs = []
    for doc in loader.lazy_load():
        docs.append(doc)
    return docs
    
async def load_url(urls):
    all_docs = []
    for url in urls:
        loader = UnstructuredLoader(web_url=url)
        docs = []
        async for doc in loader.alazy_load():
            docs.append(doc)
        all_docs.extend(docs)
    return all_docs


def load_urls_sync(urls):
    return asyncio.run(load_url(urls))
def load_all(file_path: str):
    if file_path.lower().endswith('.csv'):
        loader = CSVLoader(file_path)
    elif file_path.lower().endswith('.xlsx') or file_path.lower().endswith('.xls'):
        loader = UnstructuredExcelLoader(file_path)
    elif file_path.lower().endswith('.docx') or file_path.lower().endswith('.doc'):
        loader = UnstructuredWordDocumentLoader(file_path)
    elif file_path.lower().endswith('.pptx') or file_path.lower().endswith('.ppt'):
        loader = UnstructuredPowerPointLoader(file_path)
    elif file_path.lower().endswith('.txt'):
        loader = TextLoader(file_path)
    elif file_path.lower().endswith('.md'):
        loader = UnstructuredMarkdownLoader(file_path)
    elif file_path.lower().endswith('.json'):
        loader = JSONLoader(file_path)
    else:
        raise ValueError(f"Unsupported file type: {file_path}")

    return loader.load()

    
file_path_or_url = ["https://arxiv.org/pdf/2309.03445"]

if type(file_path_or_url) == list:
    docs = load_urls_sync(file_path_or_url)
elif file_path_or_url.lower().endswith('.pdf'):
    docs = load_pdf(file_path_or_url)
else:
    docs = load_all(file_path_or_url)

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
all_splits = text_splitter.split_documents(docs)

# Index chunks
_ = vector_store.add_documents(documents=all_splits)

prompt = hub.pull("rlm/rag-prompt")


class State(TypedDict):
    question: str
    context: List[Document]
    answer: str


# Define application steps
# def retrieve(state: State):
#     retrieved_docs = vector_store.similarity_search(state["question"])
#     return {"context": retrieved_docs}


# def generate(state: State):
#     docs_content = "\n\n".join(doc.page_content for doc in state["context"])
#     messages = prompt.invoke({"question": state["question"], "context": docs_content})
#     response = llm.invoke(messages)
#     return {"answer": response.content}


# Compile application and test
# graph_builder = StateGraph(State).add_sequence([retrieve, generate])
# graph_builder.add_edge(START, "retrieve")
# graph = graph_builder.compile()


# res =  graph.invoke({"question": "How to decide between L1 and L2 loss?"})
# print(res["answer"])

def generate(question):
    retrieved_docs = vector_store.similarity_search(question)
    docs_content = "\n\n".join(doc.page_content for doc in retrieved_docs)
    messages = prompt.invoke({"question": question, "context": docs_content})
    chunks = []
    for chunk in llm.stream(messages):
        chunks.append(chunk)
        print(chunk.content, end="", flush=True)


generate("What loss function they used in the paper?")




#############################################################################################

# prompt = ChatPromptTemplate.from_messages(
#     [
#         (
#             "system",
#             """You are a career advisor. Based on the given user profile, generate a step-by-step career progression plan.
                
#                 You will be given a user profile with the following information:
#                 User Profile: 
#                 - Interests: {{Interests}}
#                 - Skills: {{Skills}}
#                 - Learning Style: {{LearningStyle}}
#                 - Personality: {{Personality}}
#                 - Extracurricular Activities: {{Extracurricular}}
#                 - Availability: {{Availability}}
#                 - Industry Interests: {{IndustryInterests}}
#                 - Work Environment: {{WorkEnvironment}}
#                 - Education Plans: {{EducationPlans}}
#                 - Career Values: {{CareerValues}}
#                 - Preferred Job Type: {{JobType}}

#                 You have to provide a structured response as an array of steps, where each step has:
#                 1. Step number
#                 2. Title of the step
#                 3. Short description

#                 Output format(Example):
#                 [
#                     {{"step": 1, "title": "Title", "description": "Description"}},
#                     {{"step": 2, "title": "Title", "description": "Description"}}
#                 ]
                
#                 Instructions:
#                 - Only provide array, no any other things.
#                 - Try to make best possible plan based on the user profile, be specific to profile and try to make it end-to-end career progression.
#                 - Donot answer any unethical questions.
#                 """
# ,
#         ),
#         ("human", "Make a career progression for following User profile:\n{input}"),
#     ]
# )

# chain = prompt | llm


# input_msg = """
# - Interests: Programming, AI
# - Skills: Python, TensorFlow
# - Industry Interests: Technologys
# """

# chunks = []
# for chunk in chain.stream({"input": input_msg,}):
#     chunks.append(chunk)
#     print(chunk.content, end="", flush=True)