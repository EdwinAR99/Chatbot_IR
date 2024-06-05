import logging
import os

from neo4j import GraphDatabase
from retry import retry

DEPENDENCIAS_CSV_PATH = os.getenv("DEPENDENCIAS_CSV_PATH")
DIVISIONES_CSV_PATH = os.getenv("DIVISIONES_CSV_PATH")

NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s]: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

LOGGER = logging.getLogger(__name__)

NODES = ["Dependencia", "Division"]

def _set_uniqueness_constraints(tx, node):
    query = f"""CREATE CONSTRAINT IF NOT EXISTS FOR (n:{node})
        REQUIRE n.id IS UNIQUE;"""
    _ = tx.run(query, {})

@retry(tries=100, delay=10)
def load_chatbot_graph_from_csv() -> None:
    driver = GraphDatabase.driver(
        NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD)
    )

    LOGGER.info("Setting uniqueness constraints on nodes")
    with driver.session(database="neo4j") as session:
        for node in NODES:
            session.execute_write(_set_uniqueness_constraints, node)
    
    LOGGER.info("Loading dependencias nodes")
    with driver.session(database="neo4j") as session:
        query = f"""
        LOAD CSV WITH HEADERS
        FROM '{DEPENDENCIAS_CSV_PATH}' AS dependencias
        MERGE (d:Dependencia {{id: toInteger(dependencias.dependencia_id),
                            nombre: dependencias.nombre,
                            historia: dependencias.historia,
                            mision: dependencias.mision,
                            vision: dependencias.vision,
                            principios: dependencias.principios,
                            objetivos: dependencias.objetivos,
                            recepcion: dependencias.recepcion,
                            avances: dependencias.avances,
                            reintegros: dependencias.reintegros,
                            tiquetes: dependencias.tiquetes,
                            horario: dependencias.horario,
                            direccion: dependencias.direccion,
                            telefonos: dependencias.telefonos,
                            frase_vicerector: dependencias.frase_vicerector,
                            funciones: dependencias.funciones
                            }});
        """
        _ = session.run(query, {})
    
    LOGGER.info("Loading divisiones nodes")
    with driver.session(database="neo4j") as session:
        query = f"""
        LOAD CSV WITH HEADERS
        FROM '{DIVISIONES_CSV_PATH}' AS divisiones
        MERGE (div:Division {{id: toInteger(divisiones.division_id),
                            nombre: divisiones.nombre,
                            descripcion: divisiones.descripcion,
                            mision: divisiones.mision,
                            vision: divisiones.vision,
                            principios: divisiones.principios,
                            objetivos: divisiones.objetivos,
                            servicios: divisiones.servicios,
                            normatividad: divisiones.normatividad,
                            organigrama: divisiones.organigrama,
                            contacto: divisiones.contacto,
                            directorio: divisiones.directorio,
                            areas: divisiones.areas
                            }});
        """
        _ = session.run(query, {})

    LOGGER.info("Loading 'TIENE' relationships")
    with driver.session(database="neo4j") as session:
        query = f"""
        LOAD CSV WITH HEADERS FROM '{DIVISIONES_CSV_PATH}' AS divisiones
            MATCH (d:Dependencia {{id: toInteger(divisiones.dependencia_id)}})
            MATCH (div:Division {{id: toInteger(divisiones.division_id)}})
            MERGE (d)-[tiene:TIENE]->(div)
        """
        _ = session.run(query, {})

if __name__ == "__main__":
    load_chatbot_graph_from_csv()