import os

from langchain.chains import GraphCypherQAChain
from langchain.prompts import PromptTemplate
from langchain_community.graphs import Neo4jGraph
from langchain_openai import ChatOpenAI

CHATBOT_CYPHER_MODEL = os.getenv("CHATBOT_CYPHER_MODEL")
CHATBOT_QA_MODEL = os.getenv("CHATBOT_QA_MODEL")

graph = Neo4jGraph(
    url=os.getenv("NEO4J_URI"),
    username=os.getenv("NEO4J_USERNAME"),
    password=os.getenv("NEO4J_PASSWORD"),
)

graph.refresh_schema()

cypher_generation_template = """
Task:
Generate a Cypher query for a Neo4j graph database.

Instructions:
Use only the types of relationships and properties provided in the schema.
Do not use any other types of relationships or properties that are not provided.

Schema:
{schema}

Note:
Do not include explanations or apologies in your answers.
Do not answer any question that asks anything other than constructing a Cypher statement. Do not include any text except the generated Cypher statement. Ensure that the direction of the relationship is correct in your queries. Make sure to correctly alias both entities and relationships.
Do not execute any query that adds or deletes from the database.
If you need to divide numbers, ensure to filter the denominator to avoid zero.

Ejemplos:
# Listar todas las dependencias con sus divisiones.
MATCH (d:Dependencia)-[:TIENE]->(div:Division)
RETURN d.nombre AS dependencia, COLLECT(div.nombre) AS divisiones

# Que divisiones tiene la dependencia vicerectoria administrativa?.
MATCH (d:Dependencia)-[:TIENE]->(div:Division)
WHERE toLower(d.nombre) CONTAINS toLower('VICERECTORIA ADMINISTRATIVA')
RETURN div.nombre AS division

# Cuentas dependencias tiene la Universidad del Cauca? o Lista de dependecias
MATCH (d:Dependencia)
RETURN COUNT(d) AS total_dependencias

# Cuantas divisiones hay en toda la Universidad? o Lista de Divisiones
MATCH (div:Division)
RETURN COUNT(div) AS total_divisiones

# Que dependencias no tienen divisiones?
MATCH (d:Dependencia)
WHERE NOT (d)-[:TIENE]->(:Division)
RETURN d.nombre AS dependencia

# Que dependecias tiene mas divisiones en la Universidad?
MATCH (d:Dependencia)-[:TIENE]->(div:Division)
RETURN d.nombre AS dependencia, COUNT(div) AS num_divisiones
ORDER BY num_divisiones DESC
LIMIT 1

# Quiero informacion de la vicerectoria administrativa y sus divisiones.
MATCH (d:Dependencia)-[:TIENE]->(div:Division)
WHERE toLower(d.nombre) CONTAINS toLower('VICERECTORIA ADMINISTRATIVA')
RETURN d, COLLECT(div) AS divisiones

# Que divisiones tienen esta descripcion 'Descripcion especifica'?
MATCH (div:Division)
WHERE toLower(div.descripcion) CONTAINS toLower('Descripcion especifica')
RETURN div.nombre AS division, div.descripcion AS descripcion


# Muestrame todas las dependencias y divisiones de la Universidad.
MATCH (div:Division)<-[:TIENE]-(d:Dependencia)
RETURN div.nombre AS division, d.nombre AS dependencia

# Quiero las dependencias que tengan esta mision 'Mision especifica'
MATCH (d:Dependencia)-[:TIENE]->(div:Division)
WHERE toLower(div.mision) CONTAINS toLower('Mision especifica')
RETURN d.nombre AS dependencia, div.nombre AS division


Ensure to use IS NULL or IS NOT NULL when analyzing missing properties.
Never return embedding properties in your queries. Never include the "GROUP BY" statement in your query. Make sure to alias all subsequent statements as with the declaration.

The question is:
{question}
"""

cypher_generation_prompt = PromptTemplate(
    input_variables=["schema", "question"], template=cypher_generation_template
)

qa_generation_template = """You are an assistant who takes the results of a Cypher query from Neo4j and forms an understandable response for the user. The query results section contains the results of a Cypher query that was generated from the user's natural language question. The information provided is authoritative; you should never doubt it or attempt to use your internal knowledge to correct it. Make the response sound like an answer to the question.

Query Results:
{context}

Question:
{question}

If the information provided is empty, say you do not know the answer.
Empty information looks like this: []

If the information is not empty, you must provide an answer using the results. If the question implies a duration of time, assume that the query results are in units of days unless otherwise specified.

When more than one result is returned, for example, how many divisions or which divisions, you should always return a numbered list.

When names are provided in the query results, such as names of dependencies or divisions, be careful with any names that have commas or other punctuation. Ensure to return any list of names in a way that is unambiguous and allows someone to distinguish which are the complete names.

Never say you do not have the correct information if there are data in the query results. Ensure to display all relevant query results if asked.

Helpful Answer:
"""

qa_generation_prompt = PromptTemplate(
    input_variables=["context", "question"], template=qa_generation_template
)

chatbot_cypher_chain = GraphCypherQAChain.from_llm(
    cypher_llm=ChatOpenAI(model=CHATBOT_CYPHER_MODEL, temperature=0),
    qa_llm=ChatOpenAI(model=CHATBOT_QA_MODEL, temperature=0),
    graph=graph,
    verbose=True,
    qa_prompt=qa_generation_prompt,
    cypher_prompt=cypher_generation_prompt,
    validate_cypher=True,
    top_k=100,
)