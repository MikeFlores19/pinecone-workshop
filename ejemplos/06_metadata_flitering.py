import os
from dotenv import load_dotenv
from pathlib import Path
from pinecone import Pinecone

load_dotenv(Path(__file__).parent.parent / ".env")

pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

index = pc.Index("mi-primer-index")

#FETCH - traer vector por ID
print("Fetch de doc_1")
resultado=index.fetch(ids=["doc_1"])
vector=resultado.vectors["doc_1"]

print(f"ID       : doc_1")
print(f"Metadata : {vector.metadata}")
print(f"Dimensión del vector: {len(vector.values)}")

#FETCH MULTIPLE
print("\nFetch de doc_1, doc_2, doc_3 ")
resultado = index.fetch(ids=["doc_1", "doc_2", "doc_3"])
for id, vector in resultado.vectors.items():
    print(f"ID:{id} | Metadata: {vector.metadata}")

index.update(
    id="doc_1",
    set_metadata={
        "texto": "El gato duerme en el sofá (actualizado)",
        "categoria": "animales",
        "actualizado": True
    }
)

print("Después:")
print(f"{index.fetch(ids=['doc_1']).vectors['doc_1'].metadata}")


#Listar todos los IDs
print("\n Listado de todos los IDs")
resultado_ids=index.list_paginated() #crea una lista en paginas con vectors, pagination, namespace

for v in resultado_ids.vectors:
    print(f" ID: {v.id}")

print(f"\nTotal: {len(resultado_ids.vectors)} vectores")