from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy, MultipleStructuredOutputsError

from fixed.llm import chat_model
from student_parts.week02_structure_natural_language_requests import (
    StructuredRequestBatch,
    week02_system_prompt,
    week02_tools,
)

agent = create_agent(
    model=chat_model(),
    tools=week02_tools(),
    response_format=StructuredRequestBatch,
    system_prompt=week02_system_prompt(),
)

try:
    result = agent.invoke({
        "messages": [
            {"role": "user", "content": "다다음주 멘토님이랑 멘토링 월요일 5~6시 예정되어있는데 이거 예약해줘"}
        ]
    })
    print(result)
except MultipleStructuredOutputsError as exc:
    print("structured output tool call count:", len(exc.tool_names))
    print("tool names:", exc.tool_names)
    print("raw tool calls:", exc.ai_message.tool_calls)
