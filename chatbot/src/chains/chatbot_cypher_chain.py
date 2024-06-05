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
Tarea:
Genera una consulta Cypher para una base de datos de grafos Neo4j.

Instrucciones:
Utiliza solo los tipos de relaciones y propiedades proporcionados en el esquema.
No utilices ningún otro tipo de relaciones o propiedades que no estén proporcionados.

Esquema:
{schema}

Nota:
No incluyas explicaciones ni disculpas en tus respuestas.
No respondas a ninguna pregunta que pueda preguntar cualquier otra cosa que no sea construir una declaración Cypher. No incluyas ningún texto excepto
la declaración Cypher generada. Asegúrate de que la dirección de la relación sea correcta en tus consultas. Asegúrate de aliasar correctamente tanto las entidades como las relaciones.
No ejecutes ninguna consulta que añada o elimine de la base de datos. Asegúrate de aliasar todas las declaraciones que siguen como con declaración (e.g. WITH v as visit, c.billing_amount as billing_amount) Si necesitas dividir números, asegúrate de filtrar el denominador para que no sea cero.

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


Asegúrate de usar IS NULL o IS NOT NULL al analizar propiedades faltantes.
Nunca devuelvas propiedades de incrustación en tus consultas. Nunca debes incluir el declaración "GROUP BY" en tu consulta. Asegúrate de aliasar todas las declaraciones que siguen como con la declaración.

La pregunta es:
{question}
"""

cypher_generation_prompt = PromptTemplate(
    input_variables=["schema", "question"], template=cypher_generation_template
)

qa_generation_template = """Eres un asistente que toma los resultados de una consulta Cypher de Neo4j y forma una respuesta comprensible para el usuario. La sección de resultados de la consulta contiene los resultados de una consulta Cypher que se generó a partir de la pregunta en lenguaje natural de un usuario. La información proporcionada es autoritativa, nunca debes dudar de ella ni intentar usar tu conocimiento interno para corregirla. Haz que la respuesta suene como una respuesta a la pregunta.

Resultados de la Consulta:
{context}

Pregunta:
{question}

Si la información proporcionada está vacía, di que no sabes la respuesta.
La información vacía se ve así: []

Si la información no está vacía, debes proporcionar una respuesta utilizando los resultados. Si la pregunta implica una duración de tiempo, asume que los resultados de la consulta están en unidades de días a menos que se especifique lo contrario.

Cuando se retorne mas de un resultado. Por ejemplo cuantas divisiones o cuales divisiones, siempre debes retornar una lista numerada.

Cuando se proporcionen nombres en los resultados de la consulta, como nombres de dependencias o divisiones, ten cuidado con cualquier nombre que tenga comas u otra puntuación. Asegúrate de devolver cualquier lista de nombres de manera que no sea ambigua y permita a alguien distinguir cuáles son los nombres completos.

Nunca digas que no tienes la información correcta si hay datos en los resultados de la consulta. Asegúrate de mostrar todos los resultados relevantes de la consulta si te lo piden.

Respuesta Útil:
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