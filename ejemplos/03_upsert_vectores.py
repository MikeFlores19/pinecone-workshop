import os
import random
from dotenv import load_dotenv
from pathlib import Path
from pinecone import Pinecone

load_dotenv(Path(__file__).parent.parent/".env")

pc=Pinecone(api_key=os.getenv("PINECON_API_KEY"))

index=pc.Index("mi-primer-index") #Cargamos el index

#Funcion para generar un vector falso de 1536 dimensiones
def vector_random():
    return [random.uniform(-1,1) for _ in range (1536)]


# Documentos de ejemplo
documentos=[
    {
        "id": "doc_1",
        "values": vector_random(),
        "metadata": {"texto": "El gato duerme en el sofá", "categoria": "animales"}
    },
    {
        "id": "doc_2",
        "values": vector_random(),
        "metadata": {"texto": "Python es un lenguaje de programación", "categoria": "tecnologia"}
    },
    {
        "id": "doc_3",
        "values": vector_random(),
        "metadata": {"texto": "La pizza tiene queso y tomate", "categoria": "comida"}
    },
    {
        "id": "doc_4",
        "values": vector_random(),
        "metadata": {"texto": "El perro corre en el parque", "categoria": "animales"}
    },
    {
        "id": "doc_5",
        "values": vector_random(),
        "metadata": {"texto": "JavaScript se usa para web", "categoria": "tecnologia"}
    },
]

#Subir a Pinecone
index.upsert(vectors=documentos)

print(f"Subidos {len(documentos)} vectores")

#Ver estadisticas del index
stats=index.describe_index_stats() #reporte de salud/estado del index
print(f"Total vectores en el index: {stats.total_vector_count}")

