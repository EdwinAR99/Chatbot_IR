import dotenv
dotenv.load_dotenv()

from fastapi import FastAPI
from agents.chatbot_rag_agent import agent_executor, memory
from models.edu_rag_query import EduQueryInput, EduQueryOutput
from fastapi.middleware.cors import CORSMiddleware
from utils.async_utils import async_retry

app = FastAPI(
    title="Academy Chatbot",
    description="Endpoints for a education system graph RAG chatbot",
)

# Allow CORS for all origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

chat_history = memory.buffer_as_messages

#agent_executor.invoke({"input": "Hola, Buen dia! Puedes presentarte y decir cual es tu funcion todo en español", "chat_history": chat_history,})

@async_retry(max_retries=10, delay=1)
async def invoke_agent_with_retry(query: str):
    """Retry the agent if a tool fails to run.

    This can help when there are intermittent connection issues
    to external APIs.
    """
    return await agent_executor.ainvoke({"input": query, "chat_history": chat_history,})

@app.get("/")
async def get_status():
    return {"status": "running"}

@app.post("/chatbot-rag-agent")
async def query_agent(query: EduQueryInput) -> EduQueryOutput:
    query_response = await invoke_agent_with_retry(query.text)

    return query_response