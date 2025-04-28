from dotenv import load_dotenv
from langchain.tools import tool
from langchain.prompts import PromptTemplate
from langchain.tools.render import render_text_description
from langchain_openai import ChatOpenAI
from langchain.agents.output_parsers import ReActSingleInputOutputParser
from langchain.agents.format_scratchpad.log import format_log_to_str
from langchain.schema import AgentAction, AgentFinish
from langchain.tools import Tool
from typing import List, Union

@tool
def get_text_length(text: str) -> int:
    """
    Returns the length of the input text by characters.
    """
    print(f"get_text_length called with {text=}")
    text = text.strip("'\n").strip('"')
    return len(text)


def find_tool_by_name(tools: List[Tool], tool_name: str) -> Tool:
    """
    Find a tool by its name in the list of tools.
    """
    for tool in tools:
        if tool.name == tool_name:
            return tool
    raise ValueError(f"Tool with name {tool_name} not found.")
     


if __name__ == "__main__":
    load_dotenv()
    print("ReAct langchanin")

    tools = [get_text_length]

    template = """
        Answer the following questions as best you can. You have access to the following tools:

        {tools}

        Use the following format:

        Question: the input question you must answer
        Thought: you should always think about what to do
        Action: the action to take, should be one of [{tool_names}]
        Action Input: the input to the action
        Observation: the result of the action
        ... (this Thought/Action/Action Input/Observation can repeat N times)
        Thought: I now know the final answer
        Final Answer: the final answer to the original input question

        Begin!

        Question: {input}
        Thought: {agent_scratchpad}
    """

    prompt = PromptTemplate.from_template(template=template).partial(
        tools=render_text_description(tools), tool_names=", ".join([t.name for t in tools])
    )

    llm = ChatOpenAI(temperature=0, stop=["\nObservation"], model="gpt-4o-mini")
    intermediate_steps = []
    agent = (
        {"input": lambda x: x["input"], "agent_scratchpad": lambda x: format_log_to_str(x["agent_scratchpad"])}
        | prompt
        | llm
        | ReActSingleInputOutputParser()
    )
    agent_step : Union[AgentAction, AgentFinish] = agent.invoke(
        {
            "input": "What is the length of 'DOG' in char acters?",
            "agent_scratchpad": intermediate_steps
        }
    )

    print(agent_step.log)
    if isinstance(agent_step, AgentAction):
        tool_name = agent_step.tool
        tool_to_use = find_tool_by_name(tools, tool_name)
        tool_input = agent_step.tool_input
        observation = tool_to_use.func(str(tool_input))
        print(f"Observation: {observation}")
        intermediate_steps.append((agent_step, str(observation)))

    agent_step : Union[AgentAction, AgentFinish] = agent.invoke(
        {
            "input": "What is the length of 'DOG' in char acters?",
            "agent_scratchpad": intermediate_steps
        }
    )

    print(agent_step.log)
    if isinstance(agent_step, AgentFinish):
        print("Final Answer: ", agent_step.return_values['output'] )