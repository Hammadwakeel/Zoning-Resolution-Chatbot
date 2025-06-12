import os
import streamlit as st
import requests
import zipfile
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from qdrant_client import QdrantClient
from langchain import PromptTemplate
from typing_extensions import TypedDict
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, END
from langchain.schema.output_parser import StrOutputParser
from langchain_core.output_parsers import StrOutputParser
from langchain.vectorstores import Qdrant

# Streamlit app title and description
st.title("Zoning Resolution Chatbot")
st.write("Powered by LLAMA3.1 70B. Ask me anything about the Zoning Resolution!")

# Helper function to get the language model
def get_llm():
    llm = ChatGroq(
        model="llama-3.1-70b-versatile",
        temperature=0,
        max_tokens=1024,
        top_p=1,
        stream=False,
        stop=None,
        api_key='your api key'
    )
    return llm

# Helper function to get the embeddings
def get_embeddings():
    if "embeddings" not in st.session_state:
        with st.spinner("Loading embeddings..."):
            st.session_state.embeddings = HuggingFaceEmbeddings(model_name="dunzhang/stella_en_1.5B_v5")
        st.success("Embeddings loaded successfully.")
    return st.session_state.embeddings

# Helper function to get the vector store
def get_vector_store():
    if "vector_store" not in st.session_state:
        with st.spinner("Loading vector store..."):
            embeddings = get_embeddings()
            url = "https://01e87bb9-38e4-4af2-bc56-e1fa251e9888.us-east4-0.gcp.cloud.qdrant.io:6333"
            api_key = "your api key"
            client = QdrantClient(url=url, api_key=api_key, prefer_grpc=True)
            st.session_state.vector_store = Qdrant(client=client, collection_name="my_documents", embeddings=embeddings)
        st.success("Vector store loaded successfully.")
    return st.session_state.vector_store.as_retriever()

# Define the agent state class
class AgentState(TypedDict):
    question: str
    grades: list[str]
    llm_output: str
    documents: list[str]
    on_topic: bool

# Function to retrieve documents based on the question
def retrieve_docs(state: AgentState):
    question = state["question"]
    documents = retriever.get_relevant_documents(query=question)
    state["documents"] = [doc.page_content for doc in documents]
    return state

# Define the question grading class
class GradeQuestion(BaseModel):
    score: str = Field(description="Question is about Zoning Resolution? If yes -> 'Yes' if not -> 'No'")

# Function to classify the question
def question_classifier(state: AgentState):
    question = state["question"]
    system = """
    You are a grader assessing the topic of a user question.
    Only answer if the question is about one of the following topics related to zoning resolutions:
    1. Zoning laws and regulations.
    2. Land use planning and development.
    3. Zoning permits and approvals.
    4. Variances and special zoning exceptions.
    Examples: What are the zoning laws for residential areas? -> Yes
              How do I apply for a zoning variance? -> Yes
              What is the zoning for my property? -> Yes
              What is the capital of France? -> No
    If the question IS about these topics respond with "Yes", otherwise respond with "No".
    """
    grade_prompt = ChatPromptTemplate.from_messages([
        ("system", system),
        ("human", "User question: {question}"),
    ])
    llm = get_llm()
    structured_llm = llm.with_structured_output(GradeQuestion)
    grader_llm = grade_prompt | structured_llm
    result = grader_llm.invoke({"question": question})
    state["on_topic"] = result.score
    return state

# Function to route based on the topic classification
def on_topic_router(state: AgentState):
    on_topic = state["on_topic"]
    if on_topic.lower() == "yes":
        return "on_topic"
    return "off_topic"

# Function for off-topic response
def off_topic_response(state: AgentState):
    state["llm_output"] = "I can't respond to that!"
    return state

# Define the document grading class
class GradeDocuments(BaseModel):
    score: str = Field(description="Documents are relevant to the question, 'Yes' or 'No'")

# Function to grade documents based on their relevance
def document_grader(state: AgentState):
    docs = state["documents"]
    question = state["question"]
    system = """
    You are a grader assessing relevance of a retrieved document to a user question.
    If the document contains keyword(s) or semantic meaning related to the question, grade it as relevant.
    Give a binary score 'Yes' or 'No' score to indicate whether the document is relevant to the question.
    """
    grade_prompt = ChatPromptTemplate.from_messages([
        ("system", system),
        ("human", "Retrieved document: \n\n {document} \n\n User question: {question}"),
    ])
    llm = get_llm()
    structured_llm = llm.with_structured_output(GradeDocuments)
    grader_llm = grade_prompt | structured_llm
    scores = []
    for doc in docs:
        result = grader_llm.invoke({"document": doc, "question": question})
        scores.append(result.score)
    state["grades"] = scores
    return state

# Function to route based on document grades
def gen_router(state: AgentState):
    grades = state["grades"]
    if any(grade.lower() == "yes" for grade in grades):
        return "generate"
    return "rewrite_query"

# Function to rewrite the query for better results
def rewriter(state: AgentState):
    question = state["question"]
    system = """
    You are a question re-writer that converts an input question to a better version that is optimized for retrieval.
    Look at the input and try to reason about the underlying semantic intent/meaning.
    """
    re_write_prompt = ChatPromptTemplate.from_messages([
        ("system", system),
        ("human", "Here is the initial question: \n\n {question} \n Formulate an improved question."),
    ])
    llm = get_llm()
    question_rewriter = re_write_prompt | llm | StrOutputParser()
    output = question_rewriter.invoke({"question": question})
    state["question"] = output
    return state

# Function to generate the final answer
def generate_answer(state: AgentState):
    llm = get_llm()
    question = state["question"]
    context = state["documents"]
    template = """Answer the question based only on the following context:
    {context}
    Question: {question}
    """
    prompt = ChatPromptTemplate.from_template(template=template)
    chain = prompt | llm | StrOutputParser()
    result = chain.invoke({"question": question, "context": context})
    state["llm_output"] = result
    return state

# Define the workflow state graph
workflow = StateGraph(AgentState)
workflow.add_node("topic_decision", question_classifier)
workflow.add_node("off_topic_response", off_topic_response)
workflow.add_node("retrieve_docs", retrieve_docs)
workflow.add_node("rewrite_query", rewriter)
workflow.add_node("generate_answer", generate_answer)
workflow.add_node("document_grader", document_grader)
workflow.add_edge("off_topic_response", END)
workflow.add_edge("retrieve_docs", "document_grader")
workflow.add_conditional_edges("topic_decision", on_topic_router, {
    "on_topic": "retrieve_docs",
    "off_topic": "off_topic_response",
})
workflow.add_conditional_edges("document_grader", gen_router, {
    "generate": "generate_answer",
    "rewrite_query": "rewrite_query",
})
workflow.add_edge("rewrite_query", "retrieve_docs")
workflow.add_edge("generate_answer", END)
workflow.set_entry_point("topic_decision")

# Compile the workflow
app = workflow.compile()

# Load embeddings and vector store
embeddings = get_embeddings()
retriever = get_vector_store()

# Streamlit UI for chat interaction
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Input field for user question
if prompt := st.chat_input("Ask about Zoning Resolution:"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Initialize state and invoke workflow
    state = {"question": prompt}
    result = app.invoke(state)
    full_response = result["llm_output"]

    # Display response from the assistant
    with st.chat_message("assistant"):
        st.markdown(full_response)

    st.session_state.messages.append({"role": "assistant", "content": full_response})
