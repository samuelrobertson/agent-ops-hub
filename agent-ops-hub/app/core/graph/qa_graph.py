from langgraph.graph import StateGraph, END
from app.core.graph.state import State
from app.core.chains.retrieval import retrieve
from app.core.models.llm import chat_answer

def build_graph(vs):
    async def plan(state: State):
        q = state["question"]
        if q.lower().startswith("schedule"):
            return {"route":"skill"}
        return {"route":"qa"}

    async def retrieve_node(state: State):
        items = await retrieve(vs, state["question"], k=5)
        return {"retrieved": items}

    async def answer_node(state: State):
        ctx = ""
        cites = []
        for it in (state.get("retrieved") or [])[:4]:
            ctx += it["text"] + "\n"
            cites.append({"id": it["id"], "title": it["title"], "url": it.get("url"), "score": it["score"]})
        prompt = "Q: " + state["question"] + "\nContext:\n" + ctx
        ans = await chat_answer(prompt)
        return {"answer": ans, "citations": cites}

    g = StateGraph(State)
    g.add_node("plan", plan)
    g.add_node("retrieve", retrieve_node)
    g.add_node("answer", answer_node)
    g.set_entry_point("plan")
    g.add_conditional_edges("plan", lambda s: s.get("route") or "qa", {"qa":"retrieve","skill":END})
    g.add_edge("retrieve","answer")
    g.add_edge("answer", END)
    return g.compile()
