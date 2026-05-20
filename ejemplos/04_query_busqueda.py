"""
Los vectores sonr andom (no tienen significado real todavia), por lo tanto no habra coherencia semantica 
es para ver como funciona la mecánica de búsqueda
"""

import os
import random 
from dotenv import load_dotenv
from pathlib import Path
from pinecone import Pinecone

load_dotenv(Path(__file__).parent.parent/".env")

pc=Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

index=pc.Index("mi-primer-index")

#Simulamos el vector de una pregunta
vector_pregunta=[random.uniform(-1,1) for _ in range (1536)]

print("Busqueda básica (top 3)")
resultados = index.query(
    vector=vector_pregunta,
    top_k=3, #regresar los 3 mas similares
    include_metadata=True #incluir el texto y categoria que guardamos
)

for match in resultados.matches:
    print(f"\nID: {match.id}")
    print(f"Score: {match.score:.4f}") #similtud coseno (0 a 1)
    print(f"Texto: {match.metadata['texto']}")
    print(f"Categoria: {match.metadata['categoria']}")

print("\n Búsqueda con filtro por metadata")
resultados_filtrados=index.query(
    vector=vector_pregunta,
    top_k=5,
    include_metadata=True,
    filter={"categoria": {"$eq": "tecnologia"}}  # solo documentos de tecnología
)

print(f"Resultados solo de tecnología: {len(resultados_filtrados.matches)}")
for match in resultados_filtrados.matches:
    print(f"  - {match.metadata['texto']} (score: {match.score:.4f})")

    