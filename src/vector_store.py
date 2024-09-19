import os
from dotenv import load_dotenv, find_dotenv
import pandas as pd
from langchain_core.documents import Document
from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain_chroma import Chroma

_ = load_dotenv(find_dotenv())

data_path = os.getenv('DATA_PATH')

class VectorDB:
    def __init__(self, data_path: str, model_name: str, laod_data: bool = False):
        self.data_path = data_path
        self.model_name = model_name
        if laod_data:
            self.docs = self._load_data()
        self.vectorstore = self._build_vectorstore(laod_data)

    def _load_data(self):
        # Load data into a pandas DataFrame
        df = pd.read_csv(self.data_path, delimiter=';', encoding='utf-8')

        # Create documents with nutrient values as metadata
        docs = []
        for _, row in df.iterrows():
            content = f"{row['title']}; {row['description']}; {row['type']}"
            metadata = {
                'Name': row['title'],
                'Calories': float(row['calories']),
                'Fat': float(row['fat']),
                'Carbohydrates': float(row['carbs']),
                'Protein': float(row['protein']),
                'Type': row['type'],
            }
            docs.append(Document(page_content=content, metadata=metadata))
        return docs

    def _build_vectorstore(self, load_data: bool):
        # Initialize embedding function
        embedding_function = SentenceTransformerEmbeddings(model_name=self.model_name)

        # Build the vector store with documents that have metadata
        if load_data:
            vectorstore = Chroma.from_documents(
                documents=self.docs,
                embedding=embedding_function,
                persist_directory="./src/chroma_db"
            )
        else:
            vectorstore = Chroma(
                embedding_function=embedding_function,
                persist_directory="./src/chroma_db"
            )
        return vectorstore

    def similarity_search(self, query: str, constraints: dict):
        # Perform similarity search with metadata filters
        docs = self.vectorstore.similarity_search(query, filter=constraints)
        return docs