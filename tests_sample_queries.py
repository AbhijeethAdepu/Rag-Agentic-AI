import time
from src.graph import build_rag_graph

graph=build_rag_graph()

queries=[
"What is Agentic AI according to the eBook?",
"How do AI agents differ from traditional automation systems?",
"What are the core components of an Agentic Architecture?",
"What role does memory play in Agentic AI workflows?",
"Who won the 2022 FIFA World Cup?"
  ,]

for i in queries:
    r=graph.invoke({"question": i, "context": [],"answer":"","score":0.0})
    top = max((c["score"] for c in r["context"]),default=0)
    print(f"\nQ: {i}\n top similarity: {top}\n confidence:{r['score']}\n A:{r['answer'][:200]}")
    time.sleep(3)