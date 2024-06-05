#!/bin/bash

# Run any setup steps or pre-processing tasks here
echo "Running ETL to move dependencies data from csvs to Neo4j..."

# Run the ETL script
python chatbot_neo4j_etl/src/chatbot_bulk_csv_write.py