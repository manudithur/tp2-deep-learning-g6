from src.vector_store import VectorDB
import  os
from dotenv import load_dotenv, find_dotenv
from src.llm import generate_text
_ = load_dotenv(find_dotenv())

data_path = os.getenv('DATA_PATH')
embedding_model_name = os.getenv('EMBEDDING_MODEL_NAME')
vectorDB = VectorDB(data_path, embedding_model_name)

llm_model = os.getenv('LLM_MODEL')
prompt = "I want to eat fish for lunch and a plate with beef meat for dinner. What are some plates I can make?"
response = generate_text(llm_model, vectorDB.vectorstore, prompt, constraints={})

print(response)

