import time
from ragas.run_config import RunConfig
import os
from dotenv import load_dotenv
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import LLMContextRecall, Faithfulness, AnswerRelevancy, LLMContextPrecisionWithoutReference
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper

#from basic_rag_2 import create_vector_store,ask_question,create_hyde_retriever
from basic_multihop_rag import app
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
import time

load_dotenv()
hf_token=os.getenv("HUGGINGFACE_API_TOKEN") 
groq_token=os.getenv("groq")
cohere_token=os.getenv("cohere_key")

text_questions = [
    "What are the two main components combined in the Retrieval-Augmented Generation (RAG) framework?",
    
    "What is the main difference between RAG-Sequence and RAG-Token models?",
    
    "Which datasets did the paper use to evaluate open-domain question answering performance?",
    
    "What advantage does RAG have over purely parametric models like T5 or BART when world knowledge changes?",
    
    "How did RAG perform compared to BART on Jeopardy question generation according to human evaluation?"
]

ground_truths = [
    "The RAG framework combines a pre-trained seq2seq generator model (BART) as parametric memory with a dense vector index of Wikipedia accessed through a DPR neural retriever as non-parametric memory.",
    
    "RAG-Sequence uses the same retrieved document for generating the entire output sequence, while RAG-Token can use different retrieved documents for each generated token.",
    
    "The paper evaluated open-domain QA performance on Natural Questions (NQ), TriviaQA (TQA), WebQuestions (WQ), and CuratedTrec (CT).",
    
    "RAG can update its knowledge by replacing the non-parametric document index without retraining the model, whereas purely parametric models like T5 or BART require additional training to update their knowledge.",
    
    "Human evaluators found RAG generations more factual and more specific than BART generations. RAG was judged more factual in 42.7% of cases, while BART was judged more factual in only 7.1% of cases."
]

embeddings = HuggingFaceEndpointEmbeddings(model="sentence-transformers/all-MiniLM-L6-v2",huggingfacehub_api_token=hf_token)

#client=QdrantClient(path="qdrant_storage_v2")
#vector_store=QdrantVectorStore(client=client,collection_name="rag_documents_v2",embedding=embeddings)

#retriever=vector_store.as_retriever(search_type="similarity",search_kwargs={'k':2})
#llm_eval = ChatGroq(model="llama-3.1-8b-instant",api_key=groq_token,temperature=0)

#hyde_retrieve = create_hyde_retriever(vector_store, llm_eval)


answers=[]
contexts=[]


# for text_question in text_questions:
#     answer,docs=ask_question(text_question,hyde_retrieve)
#     answers.append(answer)
#     contexts.append([doc.page_content for doc in docs])
#     time.sleep(3)
for i,text_question in enumerate(text_questions):
    initial_state={"question":text_question,
        "current_question":text_question,
        "documents":[],
        "answer":"",
        "attempt_count":0,
        "is_grounded": False,
        "addresses_question":False
    }

    config={"configurable":{"thread_id":str(i)}}
    final_state=app.invoke(initial_state,config=config)
    answers.append(final_state["answer"])
    #time.sleep(10)

    docs = final_state["documents"]
    if len(docs) == 0:
        contexts.append(["No context retrieved"])
    else:
        contexts.append([doc.page_content for doc in docs])
    
    time.sleep(10)
    #contexts.append([doc.page_content for doc in final_state["documents"]])


print("Pipeline run complete")
print(f"Total answers collected: {len(answers)}")

##llm for evaluation
llm=ChatGroq(model="llama-3.3-70b-versatile",api_key=groq_token,temperature=0)
# llm = ChatGroq(model="llama-3.3-70b-versatile", api_key=groq_token, 
# temperature=0,timeout=120)

##creating object for ragas llmwrapper
ragas_llm=LangchainLLMWrapper(llm)
ragas_embeddings=LangchainEmbeddingsWrapper(embeddings)

# ragas_llm = LangchainLLMWrapper(llm)
# ragas_embeddings = LangchainEmbeddingsWrapper(embeddings)

# data = {
#     "question": text_questions,
#     "answer": answers,
#     "contexts": contexts,
#     "ground_truth": ground_truths
# }

##DATASET that contains generated vs original answers

data={
    "user_input":text_questions,
    "response":answers,
    "retrieved_contexts":contexts,
    "reference":ground_truths
}

# data = {
#     "user_input": text_questions,
#     "response": answers,
#     "retrieved_contexts": contexts,
#     "reference": ground_truths
# }

dataset = Dataset.from_dict(data)

# result = evaluate(
#     dataset=dataset,
#     metrics=[
#         LLMContextRecall(),
#         Faithfulness(),
#         AnswerRelevancy(),
#         LLMContextPrecisionWithoutReference()
#     ],
#     llm=ragas_llm,
#     embeddings=ragas_embeddings,
#     run_config=RunConfig(
#     max_workers=1,
#     timeout=180,
#     max_retries=5
# )
# )


result=evaluate(dataset=dataset,metrics=[LLMContextPrecisionWithoutReference(),Faithfulness(),AnswerRelevancy(),LLMContextRecall()],
llm=ragas_llm,embeddings=ragas_embeddings,run_config=RunConfig(max_workers=1,timeout=180,max_retries=5))


print("\n===== RAGAS EVALUATION SCORES =====")
print(result)
