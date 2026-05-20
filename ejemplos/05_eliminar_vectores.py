import os
import time
from dotenv import load_dotenv
from pathlib import Path
from pinecone import Pinecone

load_dotenv(Path(__file__).parent.parent / ".env")

pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

index = pc.Index("mi-primer-index")

# Ver todos los vectores actuales
def mostrar_vectores():
    stats = index.describe_index_stats()
    total = stats.total_vector_count
    print(f"Total vectores en el index: {total}")

    # Obtener IDs reales del index
    resultado_ids = index.list_paginated()
    ids_reales = [v.id for v in resultado_ids.vectors]

    if ids_reales:
        resultado = index.fetch(ids=ids_reales) #fetch hace la busqueda por ID's
        for id, vector in resultado.vectors.items():
            print(f"  ID: {id} | Texto: {vector.metadata['texto']}")

print(" Estado inicial")
mostrar_vectores()

# Eliminar un vector por ID
print("\n Eliminando doc_1 ")
index.delete(ids=["doc_1"])
time.sleep(1)
print("doc_1 eliminado ")
mostrar_vectores()

# Eliminar varios a la vez
print("\nEliminando doc_2 y doc_3")
index.delete(ids=["doc_2", "doc_3"])
time.sleep(1)
print("doc_2 y doc_3 eliminados")
mostrar_vectores()

# Eliminar por filtro de metadata
print("\n Eliminando todos los de categoria 'animales'")
index.delete(filter={"categoria": {"$eq": "animales"}})
time.sleep(1)
print("Categoria animales eliminada")
mostrar_vectores()

# Eliminar TODOS los vectores (descomenta para usar)

print("\nEliminando todo el index ")
index.delete(delete_all=True)
time.sleep(1)
print("Index vaciado")
mostrar_vectores()
