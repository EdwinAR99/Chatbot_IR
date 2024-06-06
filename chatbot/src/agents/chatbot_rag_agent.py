import os

from chains.chatbot_dependencia_chain import dependecia_vector_chain
from chains.chatbot_cypher_chain import chatbot_cypher_chain
from tools.formats import get_format
from langchain import hub
from langchain.agents import AgentExecutor, Tool, create_openai_functions_agent
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory

CHATBOT_AGENT_MODEL = os.getenv("CHATBOT_AGENT_MODEL")


# Herramienta adicional para manejar consultas fuera de contexto
def default_resp(question):
    return "La pregunta no está relacionada con el contexto universitario. Por favor, haz preguntas específicas sobre las dependencias de la Universidad de Cauca o los formatos asociados."

# Agent
tools = [
    Tool(
        name="Informacion",
        func=dependecia_vector_chain.invoke,
        description="""Use this when you need to answer questions about information regarding the dependencies of the University. Anything related to a specific dependency or multiple dependencies can be asked through this method. Divisions are not direct dependencies. Pass the complete question as input to the tool. For example, if the question is "What is the mission of the administrative vice-rectorate?", the response should be "What is the mission of the administrative vice-rectorate?"
        """,
    ),
    Tool(
        name="Grafos",
        func=chatbot_cypher_chain.invoke,
        description="""Use this to answer questions about the relationships between dependencies and their divisions. Use the entire prompt as input to the tool. For example, if the prompt is "What are the divisions of the Administrative Vice-Rectorate?", the input should be "What are the divisions of the Administrative Vice-Rectorate?"
        """,
    ),
    Tool(
        name="Formatos",
        func=get_format,
        description="""Use this when asked about process formats. This tool can only obtain the link to a specific process. This tool returns the link to a given process. Do not pass the word "format" or "process" as input, only the name of the format or process. For example, if the question is "What is the cancellation format?", the input should be "cancel", without accents or the root verb.
        """,
    ),
    Tool(
        name="Default",
        func=default_resp,
        description="""Use this when the question is not related to the university context."""

    ),
]

agent_prompt = hub.pull("hwchase17/openai-functions-agent")

agent_chat_model = ChatOpenAI(
    model=CHATBOT_AGENT_MODEL,
    temperature=0,
)

chatbot_agent = create_openai_functions_agent(
    llm=agent_chat_model,
    prompt=agent_prompt,
    tools=tools,
)

memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

agent_executor = AgentExecutor(
    agent=chatbot_agent,
    tools=tools,
    memory=memory,
    verbose=True,
)