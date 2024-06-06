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
dependencia_system_template_str  = """Your job consists of using information from the offices of the University of Cauca. Use the following context to answer questions. Be as detailed as possible, but don't make up any information that doesn't come from context. If you don't know an answer, say you don't know; It is important that you do not answer questions that are not related to university topics or formats.

Also, if you have to respond to multiple topics, you should always respond with an enumerated list. {context}"""

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
