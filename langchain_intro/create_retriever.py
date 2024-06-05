import dotenv
from langchain.document_loaders.csv_loader import CSVLoader
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings

CSV_PATH = "data/file.csv"
CHROMA_PATH = "chroma_data"

dotenv.load_dotenv()

loader = CSVLoader(file_path=CSV_PATH)

data = loader.load()

vector_db = Chroma.from_documents(
    data, OpenAIEmbeddings(), persist_directory=CHROMA_PATH
)