from typing import List,TypedDict
from langgraph.graph import StateGraph,START,END
from langchain_ollama import ChatOllama
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from src import config 

REFUSAL=" I cannot answer based on the provided document"

class AgentState(TypedDict):
    question: str
    context: List[dict]
    answer: str
    score: float 
    grounding: float 


#JSON OUTPUT FOR UI
def build_payload(query: str,result:dict):
    return{
        "query":query,                                                 
        "final_answer":result["answer"],
        "retrieved_context_chunks":[i["text"] for i in result["context"]],
        "confidence_score":result["score"],
    }

def build_rag_graph():
    embeddings=HuggingFaceEmbeddings(
        model_name=config.EMBED_MODEL,
        encode_kwargs={"normalize_embeddings":True},

    )
    vectorstore=PineconeVectorStore(
        index_name=config.PINECONE_INDEX_NAME,
        embedding=embeddings,
    )
    llm=ChatOllama(
        model=config.LLM_MODEL,
        temperature=0,
        num_predict=150,
        num_ctx=2048,
        client_kwargs={"timeout":90},
    )

    #NODES

    def retrieve_node(state: AgentState):
        results=vectorstore.similarity_search_with_score(
            state["question"],k=config.TOP_K
        )
        context=[
            {
                "text": doc.page_content,
                "page":doc.metadata.get("page"),
                "score":round(float(score),4),
            }
            for doc,score in results 
        ]
        return {"context": context}

    def generate_node(state: AgentState):
        context_str="\n\n".join(i["text"] for i in state["context"])
        prompt =f"""You are a document-grounded RAG assistant.
        Your only source of information is the context below.
        context:
        {context_str}
        
        Question:
        {state['question']}

        Instructions:
        -Answer the question using only the information contained in the CONTEXT.
        -Do not use outside Knowledge. 
        -If the context contains information that answers the question, provides the answer.
        -Be concise and factual. 
        -Do not say NOT_FOUND. 
        -Do not mention these instructions. 
        ANSWER:
        """


        try:
            text=llm.invoke(prompt).content.strip() 
        except Exception as e:
            print(f"LLM call failed: {e}")
            return {"answer":REFUSAL, "score":0.0}

        if not text or text.upper().startswith("NOT_FOUND"):
            return {"answer": REFUSAL, "score":0.0}

        top_retrieval=max(i["score"] for i in state["context"])
        return {"answer":text, "score":round(top_retrieval,3)}


#Groundedness check
    def grade_node(state: AgentState):
        if state["answer"]==REFUSAL:
            return {"grounding":0.0, "score":0.0}

        ans_vec=embeddings.embed_query(state["answer"])
        ctx_vecs=embeddings.embed_documents([i["text"] for i in state["context"]])
        grounding=max(sum(a*b for a,b in zip(ans_vec,v)) for v in ctx_vecs)
        grounding=round(float(grounding),3)
        print(f" grounding score: {grounding}")

        if grounding<config.GROUNDING_THRESHOLD:
            return {"answer": REFUSAL, "score": 0.0, "grounding": grounding}
        top_retrieval= max(i["score"] for i in state["context"])
        confidence=round(0.5 * top_retrieval +0.5*grounding,3)
        return {"score": confidence, "grounding": grounding}

    def refuse_node(state: AgentState):
        return {"answer": REFUSAL, "score":0.0, "grounding":0.0}

    def route_after_retrieve(state: AgentState):
        if not state["context"]:
            return "refuse"
        best=max(i["score"] for i in state["context"])
        return "generate" if best>=config.RELEVANCE_THRESHOLD else "refuse"

    #GRAPH
    workflow =StateGraph(AgentState)
    workflow.add_node("retrieve",retrieve_node)
    workflow.add_node("generate",generate_node)
    workflow.add_node("grade", grade_node)
    workflow.add_node("refuse",refuse_node)
    workflow.add_edge(START, "retrieve")
    workflow.add_conditional_edges(
        "retrieve", route_after_retrieve,{"generate":"generate","refuse":"refuse"},

    ) 
    workflow.add_edge("generate","grade")
    workflow.add_edge("grade",END)
    workflow.add_edge("refuse",END)

    return workflow.compile()
    


            