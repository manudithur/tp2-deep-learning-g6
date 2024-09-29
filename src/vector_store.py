import os
from dotenv import load_dotenv, find_dotenv
import pandas as pd
from langchain_core.documents import Document
from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain_chroma import Chroma

_ = load_dotenv(find_dotenv())

data_path = os.getenv('DATA_PATH')

class VectorDB:
    def __init__(self, data_path: str, model_name: str, load_data: bool = False):
        self.data_path = data_path
        self.model_name = model_name
        if load_data:
            self.docs = self._load_data()
        self.vectorstore = self._build_vectorstore(load_data)

    def _load_data(self):
        # Load data into a pandas DataFrame
        df = pd.read_csv(self.data_path, delimiter=';', encoding='utf-8')

        # Create documents with nutrient values as metadata
        docs = []
        for _, row in df.iterrows():
            content = f"{row['title']}; {row['description']}; {row['type']}"
            # Create type flags
            types = [t.strip() for t in row['type'].split(',')]
            type_flags = {f"Type_{t}": True for t in types}

            print(row)

            restrictions_allowed = [r.strip() for r in row['allowed'].split(',')]
            restriction_flags = {f"Allowed_{r}": True for r in restrictions_allowed}
            metadata = {
                'Source': 'food',
                'Name': row['title'],
                'Calories': float(row['calories']),
                'Fat': float(row['fat']),
                'Carbohydrates': float(row['carbs']),
                'Protein': float(row['protein']),
                **type_flags,
                **restriction_flags
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

    def similarity_search(self, query: str, count: int, constraints: dict):
        # Build filter constraints using $eq operator
        type_constraints = [
            {f"Type_{type_value}": {"$eq": True}}
            for type_value in constraints["Type"]
        ]

        restrictions_constraints = [
            {f"Allowed_{restriction_value}": {"$eq": True}}
            for restriction_value in constraints["Restrictions"]
        ]

        # Combine constraints into a single filter with a single top-level operator
        filter_constraints = {
            "$and": [
                {"$or": type_constraints},
                *restrictions_constraints  # Unpack the list of restriction constraints
            ]
        }

        # Perform similarity search with metadata filters
        docs = self.vectorstore.similarity_search(query, filter=filter_constraints, k=count)
        return docs