import os
from dotenv import load_dotenv
from pinecone import Pinecone

load_dotenv()

pc=Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

#Lista los índices existentes 
indexes=pc.list_indexes()
print("Conexión exitosa")
print(f"Índices existentes: {indexes.names()}")