from agent.tools import summary_tool, _format_context

GROUP_SIZE = 15
def summarize_document(llm,docs):
    groups = []

    for i in range(0, len(docs), GROUP_SIZE):
        groups.append(
            docs[i:i+GROUP_SIZE]
        )

    partial_summaries = []

    for group in groups:
        context = _format_context(group)
        summary = summary_tool(llm,context)
        partial_summaries.append(summary)

    combined = "\n\n".join(partial_summaries)
    final_summary = summary_tool(llm,combined)
    return final_summary