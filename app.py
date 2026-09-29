import streamlit as st
from src.graph import build_rag_graph,build_payload

st.set_page_config(page_title="RAG Chatbot", layout="wide")
@st.cache_resource(show_spinner="Loading models...")

def get_graph():
    return build_rag_graph()
graph=get_graph()

st.title("RAG Chatbot")

if "messages" not in st.session_state:
    st.session_state.messages=[]
if "last" not in st.session_state:
    st.session_state.last=None
if "payload" not in st.session_state:
    st.session_state.payload=None 

for i in st.session_state.messages:
    with st.chat_message(i["role"]):
        st.write(i["content"])

query=st.chat_input("Ask a question about the ebook")

if query:
    st.session_state.messages.append({"role":"user","content": query})
    with st.chat_message("user"):
        st.write(query)

    with st.chat_message("assistant"):
        with st.spinner("Thinking (Local model,can take a few seconds)..."):
            try:
                r=graph.invoke(
                    {
                        "question":query,
                        "context":[],
                        "answer":"",
                        "score":0.0,
                        "grounding":0.0,
                    }
                )
            except Exception as e:
                r=None
                st.error(f"Pipeline error: {e}")
        if r:
            st.write(r["answer"])
            st.session_state.messages.append(
                {"role":"assistant","content":r["answer"]}

            )
            st.session_state.last=r
            st.session_state.payload=build_payload(query,r)     

#side pannel - JSON payload 

with st.sidebar:
    st.header("Retrieval details")
    last=st.session_state.last
    if last is None:
        st.info("Ask a question to see the retrieved chunks and score")
    else:
        col1,col2=st.columns(2)
        col1.metric("Confidence", f"{last['score']:.3f}")
        col2.metric("Grounding", f"{last.get('grounding',0.0):.3f}")

        st.subheader("Retrieved chunks")
        for i,c in enumerate(last["context"],1):
            page=int(c["page"]) if c["page"] is not None else "?"
            with st.expander(f"#{i} | page {page} | similarity {c['score']}"):
                st.write(c["text"])

        st.subheader("JSON payload")
        st.json(st.session_state.payload)