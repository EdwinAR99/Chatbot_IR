import dotenv
from langchain_openai import ChatOpenAI
from langchain.prompts import (PromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate, ChatPromptTemplate, )
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts.chat import MessagesPlaceholder
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.schema.runnable import RunnablePassthrough
from langchain.agents import (
    create_openai_functions_agent,
    Tool,
    AgentExecutor,
)
from langchain import hub
from langchain.memory import ConversationBufferMemory
from langchain_intro.tools import get_format

# Cargar variables de entorno
dotenv.load_dotenv()

# Path de base de datos
REVIEWS_CHROMA_PATH = "chroma_data/"

# Procesamiento de contexto a traves de los prompt de system y humanmessage
system_template_str  = "Tu trabajo consiste en utilizar informacion de las dependecias de la Universidad de Cauca. Utiliza el siguiente contexto para responder preguntas. Se lo más detallado posible, pero no inventes ninguna información que no provenga del contexto. Si no sabes una respuesta, di que no la sabes, es importante que no respondas a preguntas que no estan relacionadas con temas universitarios o formatos. {context}"

system_prompt = SystemMessagePromptTemplate(
    prompt=PromptTemplate(
        input_variables=["context"], template=system_template_str
    )
)

human_prompt = HumanMessagePromptTemplate(
    prompt=PromptTemplate(
        input_variables=["question"], template="{question}",
    )
)

messages = [system_prompt, human_prompt,]
review_prompt_template = ChatPromptTemplate(
    input_variables=["context","question"], messages=messages,
)

# Intancia del modelo (GPT 3.5 en este caso)
chat_model = ChatOpenAI(model="gpt-3.5-turbo-0125", temperature=0)

# Formateador de salida
output_parser = StrOutputParser()

# Recuperador de reviews
vector_db = Chroma(
    persist_directory=REVIEWS_CHROMA_PATH,
    embedding_function=OpenAIEmbeddings()
)

retriever  = vector_db.as_retriever(k=10)

# Concatenacion de caracteristicas de todo el modelo
chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | review_prompt_template 
    | chat_model 
    | output_parser
)

# Herramienta adicional para manejar consultas fuera de contexto
def default_resp(question):
    return "La pregunta no está relacionada con el contexto universitario. Por favor, haz preguntas específicas sobre las dependencias de la Universidad de Cauca o los formatos asociados."


# Agent tools
tools = [
    Tool(
        name="Informacion",
        func=chain.invoke,
        description="""Úsalo cuando necesites responder preguntas sobre informacion de las dependencias de la Universidad como historia, mision, vision, principios, objetivos, recepcion(solicitudes que se reciben en vicerectoria), avances(solicitudes de avances), reintegros, tiquetes, horario, direccion, telefonos, frase del vicerector, funciones. No es útil para responder preguntas sobre formatos. Pase la pregunta completa como entrada a la herramienta. Por ejemplo, si la pregunta es "¿Cual es la mision de la vicerectoria administrativa?", la respuesta debe ser "¿Cual es la mision de la vicerectoria administrativa?"
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
    model="gpt-3.5-turbo-1106",
    temperature=0,
)

agent = create_openai_functions_agent(
    llm=agent_chat_model,
    prompt=agent_prompt,
    tools=tools,
)

memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    memory=memory,
    verbose=True,
)