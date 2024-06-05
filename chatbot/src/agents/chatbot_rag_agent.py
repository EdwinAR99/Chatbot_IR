import os

from chatbot.src.chains.chatbot_dependencia_chain import dependecia_vector_chain
from chatbot.src.chains.chatbot_cypher_chain import chatbot_cypher_chain
from chatbot.src.tools.formats import get_format
from langchain import hub
from langchain.agents import AgentExecutor, Tool, create_openai_functions_agent
from langchain_openai import ChatOpenAI
from langchain_intro.tools import get_format
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
        description="""Úsalo cuando necesites responder preguntas sobre informacion de las dependencias de la Universidad. Todo lo relacionado con una dependencia en especifico y varias, las puedes preguntar por este metodo. Las divisiones no son dependencias directas. Pase la pregunta completa como entrada a la herramienta. Por ejemplo, si la pregunta es "¿Cual es la mision de la vicerectoria administrativa?", la respuesta debe ser "¿Cual es la mision de la vicerectoria administrativa?"
        """,
    ),
    Tool(
        name="Grafos",
        func=chatbot_cypher_chain.invoke,
        description="""Úsalo para responder preguntas sobre las relaciones entre dependencias y sus divisiones. Utiliza todo el prompt como entrada para la herramienta. Por ejemplo, si el prompt es "¿Cuales son las divisiones de la Vicerectoria Administrativa?", la entrada debe ser "¿Cuales son las divisiones de la Vicerectoria Administrativa?"
        """,
    ),
    Tool(
        name="Formatos",
        func=get_format,
        description="""Úsalo cuando se le pregunte sobre formatos de procesos. Esta herramienta solo puede obtener el enlace de un proceso en especifico. Esta herramienta devuelve el enlace de un determinado proceso. No pase la palabra "formato" o "proceso" como entrada, solo el nombre del formato o proceso. Por ejemplo, si la pregunta es "¿Cuál es el formato de cancelación?", la entrada debe ser "cancelar", sin tildes o el verbo raiz.
        """,
    ),
    Tool(
        name="Default",
        func=default_resp,
        description="Úsalo cuando la pregunta no esté relacionada con el contexto universitario."

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