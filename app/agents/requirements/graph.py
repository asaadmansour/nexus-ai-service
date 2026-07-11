from langgraph.graph import END, START, StateGraph

from app.agents.requirements.nodes import (
    check_missing_fields_node,
    choose_next_question_node,
    extract_requirements_node,
    merge_brief_node,
    prepare_brief_context_node,
)
from app.agents.requirements.state import RequirementsState


def build_requirements_graph():
    graph = StateGraph(RequirementsState)

    graph.add_node("prepare_brief_context", prepare_brief_context_node)
    graph.add_node("extract_requirements", extract_requirements_node)
    graph.add_node("merge_brief", merge_brief_node)
    graph.add_node("check_missing_fields", check_missing_fields_node)
    graph.add_node("choose_next_question", choose_next_question_node)

    graph.add_edge(START, "prepare_brief_context")
    graph.add_edge("prepare_brief_context", "extract_requirements")
    graph.add_edge("extract_requirements", "merge_brief")
    graph.add_edge("merge_brief", "check_missing_fields")
    graph.add_edge("check_missing_fields", "choose_next_question")
    graph.add_edge("choose_next_question", END)

    return graph.compile()


requirements_graph = build_requirements_graph()
