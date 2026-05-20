import os
from dotenv import load_dotenv
from pathlib import Path
from pinecone import Pinecone, ServerlessSpec

load_dotenv(Path(__file__).parent.parent/".env") #abre .env y carga PINECONE_API_KEY en os.environ

pc=Pinecone(api_key=os.getenv("PINECONE_API_KEY")) #hace la conexion a pinecone con el API

INDEX_NAME="mi-primer-index"

#Verificar si ya existe par ano duplicar
if INDEX_NAME not in pc.list_indexes().names():
    pc.create_index(
        name=INDEX_NAME,
        dimension=1536, #debe ser compatible con el modelo en cuestion de dimension en este caso con text-embedding-3-small de OpenAI
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1" #region gratuita - plan free
        )
    )
    print(f"Index '{INDEX_NAME}' creado")
else:
    print(f"Index '{INDEX_NAME}' ya existía")

#Detalles del index
index_info=pc.describe_index(INDEX_NAME)
print(f"\nDetalles:")
print(f"Nombre: {index_info.name}")
print(f"Dimensión: {index_info.dimension}")
print(f"Métrica: {index_info.metric}")
print(f"Estado: {index_info.status.ready}")


