import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from typing import TypedDict,List

#langgrapgh
from langgraph.graph import StateGraph,START,END
from langgraph.checkpoint.memory import MemorySaver
import cohere
import json

from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers.ensemble import EnsembleRetriever

#load api keys
load_dotenv()
hf_token = os.getenv("HUGGINGFACE_API_TOKEN")
groq_token = os.getenv("groq")
cohere_token = os.getenv("cohere_key")

COLLECTION_NAME = "rag_documents_v2"
QDRANT_PATH = "qdrant_storage_v2"
ENABLE_RERANKING = True
MAX_RETRIES = 3

llm = ChatGroq(model="llama-3.1-8b-instant", api_key=groq_token, temperature=0)
output_parser = StrOutputParser()

embeddings = HuggingFaceEndpointEmbeddings(model="sentence-transformers/all-MiniLM-L6-v2",huggingfacehub_api_token=hf_token)

client = QdrantClient(path=QDRANT_PATH)
vector_store = QdrantVectorStore(client=client,collection_name=COLLECTION_NAME,embedding=embeddings)

#retriever = vector_store.as_retriever(search_type="similarity",search_kwargs={"k":5})

def create_hybrid_retriever():
    all_docs = vector_store.similarity_search("", k=1000)
    texts = [doc.page_content for doc in all_docs]
    
    bm25_retriever = BM25Retriever.from_texts(texts)
    bm25_retriever.k = 5
    
    dense_retriever = vector_store.as_retriever(search_type="similarity",search_kwargs={"k": 4})
    
    ensemble = EnsembleRetriever(
        retrievers=[bm25_retriever, dense_retriever],
        weights=[0.3, 0.7])
    
    return ensemble

hybrid_retriever = create_hybrid_retriever()

class RAGState(TypedDict):
    question: str
    current_question: str
    documents: List[Document]
    answer: str
    attempt_count: int
    is_grounded: bool
    addresses_question: bool

#Creating 

#query enrichment
def enrich_query(query: str):

    #short_query = len(query.split()) <= 4

    if query:
        return f"""
        {query}

        In the context of AI machine learning NLP transformers
        retrieval augmented generation
        large language models
        """

    return query

#1) Retrieval Node
def retrieve_node(state:RAGState)->dict:
    print("---retrieve node--")
    
    current_question=state["current_question"]
    #documents=retriever.invoke(current_question)
    query = enrich_query(current_question)

    documents = hybrid_retriever.invoke(query)
    print(f"Retrieved {len(documents)} chunks")

    return {"documents":documents,"attempt_count":state["attempt_count"]+1}

#2) Grade Dcocument Node
def grade_documents_node(state:RAGState)->dict:
    print("---grade node--")

    question=state["question"]
    documents=state["documents"]

    grade_prompt = PromptTemplate(
    template="""You are a document relevance classifier.Your task is to determine whether the retrieved document contains information that helps answer the user's question.
    Rules:
    - Respond ONLY with valid JSON.
    - Classify or score yes only when retrieved document has artificial intelligence,machine learning,natural language processing,deep learning related words.
    - Do NOT explain your reasoning.
    - Do NOT summarize the document.
    - Do NOT output anything except the JSON object.
    - Output must be exactly one of: {{"score":"yes"}}
    or {{"score":"no"}} Question:{question}
    Document:{document}Is this document relevant to the question?""",input_variables=["document", "question"])

    relevant_documents=[]

    for doc in documents:
        chain= grade_prompt | llm | output_parser
        results=chain.invoke({"document":doc.page_content,"question":question})

        try:
            clean = results.strip()
            clean = clean.replace("```json", "").replace("```", "").strip()
            score=json.loads(clean)

            if score["score"]=='yes':
                relevant_documents.append(doc)
                print("Document graded: RELEVANT")
            else:
                print("Document graded: NOT RELEVANT, discarding")
        except:
            relevant_documents.append(doc)
            print("Grading parse failed, keeping document")

    print(f"relevance documents length after grading{len(relevant_documents)}")  
    return {"documents": relevant_documents}

    #3) Reformulate query node
def reformulate_node(state: RAGState) -> dict:
    print(f"\n--- REFORMULATE NODE ---")
    
    current_question = state["current_question"]
    reformulate_prompt = PromptTemplate(template="""You are an expert query reformulation systemfor AI/ML/NLP research papers.Your job is ONLY to improve retrieval qualitywhile preserving the original meaning.

STRICT RULES:
1. NEVER change the domain of the question.
2. NEVER invent new meanings for acronyms.
3. NEVER infer acronym meanings from weak retrievals.
4. If acronym meaning is unclear, preserve the acronym unchanged.
5. Keep the reformulated query semantically close to original query.
6. Do not introduce unrelated fields like
   project management, enterprise systems,
   healthcare, finance, etc.
7. Assume questions belong to AI/ML/NLP domain unless explicitly stated otherwise.

Original Question:{question} Return ONLY the rewritten query.""",input_variables=["question"])
    
    chain = reformulate_prompt | llm | output_parser
    new_question = chain.invoke({"question": current_question})
    
    print(f"Original query: {current_question}")
    print(f"Reformulated query: {new_question}")
    
    return {"current_question": new_question}              


#4) Rerank node

def rerank_node(state: RAGState) -> dict:
    print(f"\n--- RERANK NODE ---")
    
    question = state["current_question"]
    documents = state["documents"]
    
    if not ENABLE_RERANKING or not cohere_token:
        print("Reranking disabled, skipping")
        return {"documents": documents}
    
    try:
        co = cohere.ClientV2(api_key=cohere_token)
        passages = [doc.page_content for doc in documents]
        
        response = co.rerank(model="rerank-v3.5",query=question,documents=passages,top_n=2)
        
        reranked_documents = [documents[result.index] for result in response.results]
        print(f"Reranked documents, keeping top 2")
        return {"documents": reranked_documents}
    
    except Exception as e:
        print(f"Reranking failed: {e}, keeping original documents")
        return {"documents": documents}


#Answer Generate Node
def generate_node(state: RAGState) -> dict:
    print(f"\n--- GENERATE NODE ---")
    
    question = state["current_question"]
    documents = state["documents"]
    
    context = " ".join([doc.page_content for doc in documents])
  
    prompt = PromptTemplate(
    template="""You are a grounded AI research assistant.Your task is to answer the user's question ONLY using the provided context from AI/ML/NLP research documents.
    ACRONYM RESOLUTION RULES:
1. If an acronym appears in the question or context:
   - Search carefully for its explicit definition in the provided context.
   - Look for patterns like:
     - "Full Form (ACRONYM)"
     - "ACRONYM stands for ..."
     - first introduction of the acronym

2. ONLY expand an acronym if its meaning is explicitly supported by the retrieved context.

3. NEVER invent or guess acronym meanings from prior knowledge.

4. If multiple meanings exist:
   - choose the meaning most consistent with AI/ML/NLP research context.

5. If the acronym meaning is unclear:
   - keep the acronym unchanged instead of hallucinating.

ANSWERING RULES:

1. Answer ONLY from the provided context.

2. Use simple, natural language.

3. Do not copy large portions of text verbatim.

4. Do not introduce outside knowledge.

5. If the answer cannot be determined from the context, say:
   "I don't know based on the provided documents."

6. Keep answers concise but informative. Answers should be atleast five lines.

CONTEXT:{context},QUESTION:{question}.Answer:""",input_variables=["context", "question"])
    chain = prompt | llm | output_parser
    answer = chain.invoke({"context": context, "question": question})
    
    print(f"Answer generated")
    return {"answer": answer}

#Hallucination Check
def hallucination_check_node(state: RAGState) -> dict:
    print(f"\n--- HALLUCINATION CHECK NODE ---")
    
    answer = state["answer"]
    documents = state["documents"]
    
    context = " ".join([doc.page_content for doc in documents])
    
    hallucination_prompt = PromptTemplate(
        template="""You are a grader checking if an answer is grounded in the provided context.
        Score yes only when retrieved document has artificial intelligence,machine learning,natural language processing,deep learning related words.

    Context: {context}
    Answer: {answer}
    Check if the answer is fully supported by the context.
    You must respond with ONLY the JSON object. No other text.
    Example: {{"score": "yes"}}""",input_variables=["context", "answer"])
    
    chain = hallucination_prompt | llm | output_parser
    result = chain.invoke({"context": context,"answer": answer})
    
    try:
        score = json.loads(result)
        is_grounded = score["score"] == "yes"
        print(f"Hallucination check: {'PASSED' if is_grounded else 'FAILED'}")
        return {"is_grounded": is_grounded}
    except:
        print("Hallucination parse failed, assuming grounded")
        return {"is_grounded": True}

##Answer Grader
def answer_grade_node(state: RAGState) -> dict:
    print(f"\n--- ANSWER GRADE NODE ----")
    
    question = state["current_question"]
    answer = state["answer"]
    
    answer_prompt = PromptTemplate(
        template="""You are a grader checking if an answer addresses the question asked.Question: {question} Answer: {answer} Check if the answer actually addresses what was asked in the question.
    You must respond with ONLY the JSON object. No other text.
    Example: {{"score": "yes"}}""",input_variables=["question", "answer"])
    
    chain = answer_prompt | llm | output_parser
    result = chain.invoke({"question": question, "answer": answer})
    
    try:
        score = json.loads(result)
        addresses = score["score"] == "yes"
        print(f"Answer grade: {'ADDRESSES QUESTION' if addresses else 'DOES NOT ADDRESS QUESTION'}")
        return {"addresses_question": addresses}
    except:
        print("Answer grade parse failed, assuming good answer")
        return {"addresses_question": True}


def should_retrieve_or_end(state: RAGState) -> str:
    if state["attempt_count"] >= MAX_RETRIES:
        print("Max retries reached, ending")
        return "end"
    return "retrieve"

def should_generate_or_reformulate(state: RAGState) -> str:
    if len(state["documents"]) == 0:
        print("No relevant documents, reformulating")
        return "reformulate"
    return "rerank"

def should_retry_or_end_after_hallucination(state: RAGState) -> str:
    if not state["is_grounded"]:
        print("Answer not grounded, reformulating")
        return "reformulate"
    return "answer_grade"

def should_finish_or_reformulate(state: RAGState) -> str:
    if not state["addresses_question"]:
        print("Answer does not address question, reformulating")
        return "reformulate"
    return "end"


memory = MemorySaver()

graph = StateGraph(RAGState)

graph.add_node("retrieve", retrieve_node)
graph.add_node("grade_documents", grade_documents_node)
graph.add_node("reformulate", reformulate_node)
graph.add_node("rerank", rerank_node)
graph.add_node("generate", generate_node)
graph.add_node("hallucination_check", hallucination_check_node)
graph.add_node("answer_grade", answer_grade_node)

graph.add_edge(START, "retrieve")
graph.add_edge("retrieve", "grade_documents")

graph.add_conditional_edges("grade_documents",should_generate_or_reformulate,{"reformulate": "reformulate", "rerank": "rerank"})

graph.add_edge("rerank", "generate")

graph.add_edge("generate", "hallucination_check")

graph.add_conditional_edges("hallucination_check",should_retry_or_end_after_hallucination,{"reformulate": "reformulate", "answer_grade": "answer_grade"})

graph.add_conditional_edges("answer_grade",should_finish_or_reformulate,{"reformulate": "reformulate", "end": END})

graph.add_conditional_edges("reformulate",should_retrieve_or_end,{"retrieve": "retrieve", "end": END})

app = graph.compile(checkpointer=memory)
print("Graph compiled successfully")


if __name__ == "__main__":
    
    question = "What is RAG? Answer in six lines."
    
    initial_state = {"question": question,"current_question": question,"documents": [],"answer": "","attempt_count": 0,"is_grounded": False,"addresses_question": False}
    
    config = {"configurable": {"thread_id": "1"}}
    
    print(f"Question: {question}")
    print("\nStarting LangGraph pipeline...\n")
    
    final_state = app.invoke(initial_state, config=config)
    
    print(f"\nFinal Answer: {final_state['answer']}")
    print(f"Total attempts: {final_state['attempt_count']}")
    print(f"Is grounded: {final_state['is_grounded']}")
    print(f"Addresses question: {final_state['addresses_question']}")

    client.close()

