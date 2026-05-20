import os
import time
import fitz #pymupdf
from dotenv import load_dotenv
from pathlib import Path
from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer
import ollama


load_dotenv(Path(__file__).parent.parent/".env")

#CONFIGURACION
pc=Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

modelo_embeddings=SentenceTransformer("paraphrase-multilingual-mpnet-base-v2")


INDEX_NAME="rag-nutricion"
PDF_PATH=Path(__file__).parent.parent/"data"/"guia_nutricion.pdf"

#PASO 1 CREAR INDEX
def crear_index():
    if INDEX_NAME not in pc.list_indexes().names():
        pc.create_index(
            name=INDEX_NAME,
            dimension=768,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws",region="us-east-1")
        )
        time.sleep(4)
        print(f"Index {INDEX_NAME} creado")
    else:
        print(f"Index {INDEX_NAME} ya existe")
    return pc.Index(INDEX_NAME)

#PASO 2 LEER PDF Y HACER CHUNKS
def extraer_chunks(pdf_path, chunk_size=800,overlap=150): #500 caracteres, toma los ultimos 50 dle anterior para tener contexto
    doc=fitz.open(pdf_path)
    chunks=[]

    for num_pagina, pagina in enumerate(doc):
        texto_pagina=pagina.get_text()
        inicio=0
        while inicio<len(texto_pagina):
            fin=inicio+chunk_size
            chunk=texto_pagina[inicio:fin].strip()
            if chunk:
                chunks.append({
                    "texto":chunk,
                    "pagina":num_pagina + 1,
                    "documento":pdf_path.name
                })
            inicio+=chunk_size-overlap

        
    print(f"PDF procesao: {len(chunks)} chunks extraidos")

    return chunks


#PASO 3 SUBIR CHUNKS A PINECONE
def indexar_chunks(index,chunks):
    print("Generando embeddings y subiendo a Pinecone...")
    vectores=[]

    for i, chunk in enumerate(chunks):
        embedding = modelo_embeddings.encode(chunk["texto"]).tolist()
        vectores.append({
            "id": f"chunk_{i}",
            "values": embedding,
            "metadata": {
                "texto": chunk["texto"],
                "pagina": chunk["pagina"],
                "documento": chunk["documento"],
                "chunk_id": i
            }
        })

    #Subir en ltoes de 50
    lote_size=50

    for i in range(0,len(vectores),lote_size):
        lote=vectores[i:i+lote_size]
        index.upsert(vectors=lote)
        print(f"Subidos chunks {i} - {i+len(lote)}")

    time.sleep(5)
    stats=index.describe_index_stats()
    print(f"Total chunks en Pinecone. {stats.total_vector_count}")


#PASO 4 RAG - PREGUNTAR
def preguntar (index,pregunta):
    print(f"\nPregunta: {pregunta}")
    
    embedding_pregunta=modelo_embeddings.encode(pregunta).tolist() #convierte la pregunta al mismo dominio de embeddings

    #Buscar chunks rlevantes en  Pinecone
    resultados=index.query(
        vector=embedding_pregunta,
        top_k=5,
        include_metadata=True
    )

    print("Fuentes encontradas:")
    for i, match in enumerate(resultados.matches):
        print(f"Fuente {i+1}: {match.metadata['documento']} | "
        f"Página {match.metadata['pagina']} | "
        f"Score: {match.score:.3f}")

    #Constuir contexto con los chunks encontrados
    contexto="\n\n".join([
        f"[Fragmento {i+1}]: {match.metadata['texto']}"
        for i, match in enumerate (resultados.matches)
    ]) #une los fragmentos a partir de los 3 matches mas similares a la pregunta , extrayendo el texto

    #Enviar a Gemini con el contexto
    prompt = f"""Eres un asistente experto en nutrición.
    Responde SIEMPRE en español.
    Responde la pregunta basándote ÚNICAMENTE en el siguiente contexto extraído de una guía de nutrición.
    Si la respuesta no está en el contexto, dilo claramente.

    CONTEXTO:
    {contexto}

    PREGUNTA: {pregunta}

    RESPUESTA:"""

    respuesta=ollama.chat(
        model="qwen2.5:7b",
        messages=[{"role":"user","content":prompt}]
    )
    print(f"Respuesta:{respuesta['message']['content']}") #se le manda el prompt al agente


#MAIN

if __name__ == "__main__":
    # Crear index
    index=crear_index()

    # Procesar PDF y subir (solo si está vacío)
    stats=index.describe_index_stats()
    if stats.total_vector_count == 0:
        chunks=extraer_chunks(PDF_PATH)
        indexar_chunks(index, chunks)
    else:
        print(f"Index ya tiene {stats.total_vector_count} chunks, saltando indexación")

    # Hacer preguntas
    preguntar(index,"¿Cuánta proteína necesito al día?")
    preguntar(index,"¿Qué alimentos tienen más calcio?")
    preguntar(index,"¿Es malo comer carbohidratos?")
    preguntar(index, "¿Cuáles son los mejores ejercicios para bajar de peso?")
