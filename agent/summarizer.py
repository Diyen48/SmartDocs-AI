from agent.tools import summary_tool, _format_context

GROUP_SIZE = 15

def summarize_document(llm, docs):

    if len(docs) <= GROUP_SIZE:
        return summary_tool(
            llm,
            _format_context(docs)
        )
    groups = []
    for i in range(0, len(docs), GROUP_SIZE):
        groups.append(docs[i:i+GROUP_SIZE])

    partial_summaries = []

    for group in groups:

        summary = summary_tool(
            llm,
            _format_context(group)
        )

        partial_summaries.append(summary)

    combined = "\n\n".join(partial_summaries)

    return summary_tool(
        llm,
        combined,
    )