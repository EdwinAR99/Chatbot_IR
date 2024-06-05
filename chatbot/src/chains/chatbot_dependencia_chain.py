import os

from langchain.chains import RetrievalQA
from langchain.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    PromptTemplate,
    SystemMessagePromptTemplate,
)
from langchain.vectorstores.neo4j_vector import Neo4jVector
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

CHATBOT_QA_MODEL = os.getenv("CHATBOT_QA_MODEL")

neo4j_vector_index = Neo4jVector.from_existing_graph(
    embedding=OpenAIEmbeddings(),
    url=os.getenv("NEO4J_URI"),
    username=os.getenv("NEO4J_USERNAME"),
    password=os.getenv("NEO4J_PASSWORD"),
    index_name="dependencias",
    node_label="Dependencia",
    text_node_properties=[
        "nombre",
        "historia",
        "mision",
        "principios",
        "objetivos",
        "recepcion",
        "avances",
        "reintegros",
        "tiquetes",
        "horario",
        "direccion",
        "telefonos",
        "frase_vicerector",
        "funciones"
    ],
    embedding_node_property="embedding",
)

# Procesamiento de contexto a traves de los prompt de system y humanmessage
dependencia_system_template_str  = """Tu trabajo consiste en utilizar información de las dependencias de la Universidad de Cauca. Utiliza el siguiente contexto para responder preguntas. Sé lo más detallado posible, pero no inventes ninguna información que no provenga del contexto. Si no sabes una respuesta, di que no la sabes; es importante que no respondas a preguntas que no están relacionadas con temas universitarios o formatos.

Además, en caso de que debas responder a múltiples temas, siempre debes responder con una lista enumerada. {context}"""

dependencia_system_prompt = SystemMessagePromptTemplate(
    prompt=PromptTemplate(
        input_variables=["context"], template=dependencia_system_template_str
    )
)

dependencia_human_prompt = HumanMessagePromptTemplate(
    prompt=PromptTemplate(
        input_variables=["question"], template="{question}",
    )
)

messages = [dependencia_system_prompt, dependencia_human_prompt,]
dependencia_prompt = ChatPromptTemplate(
    input_variables=["context","question"], messages=messages,
)

# Concatenacion de caracteristicas de todo el modelo
dependecia_vector_chain = RetrievalQA.from_chain_type(
    llm=ChatOpenAI(model=CHATBOT_QA_MODEL, temperature=0),
    chain_type="stuff",
    retriever=neo4j_vector_index.as_retriever(k=12),
)

dependecia_vector_chain.combine_documents_chain.llm_chain.prompt = dependencia_prompt
