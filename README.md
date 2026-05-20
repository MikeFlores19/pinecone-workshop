# 🌲 Workshop de Pinecone — Bases de Datos Vectoriales y RAG

Aprende a usar Pinecone desde cero: desde la conexión básica hasta construir un sistema RAG completo con un PDF real en español.

> ⚠️ **Importante:** Este proyecto usa una API key de Pinecone de una cuenta específica. Para correrlo tú mismo necesitas crear tu propia cuenta en [pinecone.io](https://www.pinecone.io) y agregar tus propias credenciales en el archivo `.env`. Ver sección [Setup](#setup).

> 🤖 **LLM usado actualmente:** El ejemplo 07 corre con **Ollama** usando el modelo **qwen2.5:7b** de forma local. Puedes cambiarlo por Gemini, OpenAI u otro modelo — ver sección [Cambiar el LLM](#cambiar-el-llm).

---

## 📁 Estructura del proyecto

```
dev_pinecone/
├── .env                          # API keys 
├── .gitignore
├── requirements.txt
├── README.md
├── data/
│   └── guia_nutricion.pdf        # PDF usado para el RAG
└── ejemplos/
    ├── 01_conexion.py            # Conexión básica a Pinecone
    ├── 02_crear_index.py         # Crear un index
    ├── 03_upsert_vectores.py     # Subir vectores con metadata
    ├── 04_query_busqueda.py      # Búsqueda semántica y filtros
    ├── 05_eliminar_vectores.py   # Eliminar vectores
    ├── 06_metadata_filtering.py  # Fetch, update y list
    └── 07_rag_completo.py        # RAG completo con PDF + LLM
```

---

## ⚙️ Setup

### 1. Requisitos

- Python 3.10+
- Cuenta gratuita en [Pinecone](https://www.pinecone.io)
- [Ollama](https://ollama.com) instalado localmente con el modelo `qwen2.5:7b`

### 2. Clonar e instalar dependencias

```bash
git clone https://github.com/MikeFlores19/pinecone-workshop.git
cd pinecone-workshop

python3 -m venv venv
source venv/bin/activate        # Mac/Linux
# venv\Scripts\activate         # Windows

pip install pinecone python-dotenv sentence-transformers pymupdf ollama
```

### 3. Configurar variables de entorno

Crea un archivo `.env` en la raíz del proyecto con tus propias credenciales:

```
PINECONE_API_KEY=pcsk_xxxxxxxxxx     # Tu API key de Pinecone
```

> El `.env` está en `.gitignore` — nunca se sube a GitHub.

Para obtener tu API key de Pinecone:
1. Ve a [pinecone.io](https://www.pinecone.io) y crea una cuenta gratuita
2. En el dashboard ve a **API Keys** en el menú lateral
3. Copia la key que empieza con `pcsk_...`

### 4. Instalar Ollama y el modelo

```bash
# Instalar Ollama (Linux)
curl -fsSL https://ollama.com/install.sh | sh

# Descargar el modelo
ollama pull qwen2.5:7b
```

### 5. Verificar conexión

```bash
python ejemplos/01_conexion.py
```

Deberías ver:
```
Conexión exitosa ✅
Índices existentes: []
```

---

## 🧠 ¿Qué es Pinecone?

Pinecone es una **base de datos vectorial** administrada (serverless). En lugar de guardar filas con texto como una base de datos tradicional, guarda vectores numéricos de alta dimensión que representan el *significado semántico* de un texto, imagen o audio.

### ¿Qué es un vector?

Cuando un modelo de embeddings convierte texto en vector, produce una lista fija de números. El tamaño de esa lista es la **dimensión**:

```python
"El gato duerme"      → [0.21, -0.87, 0.54, ...]  # 768 números
"Python es genial"    → [0.63,  0.12, -0.34, ...]  # 768 números
"La pizza tiene queso"→ [-0.12, 0.45, 0.91, ...]   # 768 números
```

Siempre el mismo tamaño sin importar el texto. Los **valores** cambian según el significado — frases similares producen vectores parecidos.

### Diferencia con búsqueda tradicional

| Búsqueda tradicional (SQL) | Búsqueda vectorial (Pinecone) |
|---------------------------|-------------------------------|
| `WHERE texto LIKE '%perro%'` | Busca por significado |
| Solo encuentra coincidencias exactas | Encuentra "can", "mascota", "cachorro" |
| No entiende sinónimos | Entiende contexto semántico |

### Modelos de embeddings y sus dimensiones

| Modelo | Dimensión | Idiomas |
|--------|-----------|---------|
| `paraphrase-multilingual-mpnet-base-v2` | **768** ← usado en este proyecto | ✅ Español y 50+ idiomas |
| `all-mpnet-base-v2` | 768 | Inglés |
| `all-MiniLM-L6-v2` | 384 | Inglés |
| `text-embedding-3-small` (OpenAI) | 1536 | Multilingüe |

> ⚠️ La dimensión del index en Pinecone **debe coincidir exactamente** con el modelo de embeddings. No se puede cambiar después de crear el index.

---

## 📚 Conceptos clave

### Index

Equivalente a una tabla en SQL, pero para vectores. Se define con:

```python
pc.create_index(
    name="mi-index",
    dimension=768,        # debe coincidir con el modelo de embeddings
    metric="cosine",      # forma de medir similitud
    spec=ServerlessSpec(
        cloud="aws",
        region="us-east-1"  # región gratuita en plan free
    )
)
```

### Métricas de similitud

| Métrica | Cuándo usar |
|---------|-------------|
| `cosine` | Texto y semántica — el más común |
| `dotproduct` | Cuando los vectores están normalizados |
| `euclidean` | Datos numéricos continuos |

### Score de similitud

Con `metric="cosine"` Pinecone devuelve un score entre 0 y 1:

```
0.90+ → muy similar     "perro corre" vs "can trota"
0.70  → algo similar
0.50  → poco relacionado
0.40- → sin relación     (como en preguntas fuera del PDF)
```

### Operaciones principales

| Operación | Método | ¿Qué hace? |
|-----------|--------|------------|
| Subir vectores | `index.upsert()` | Inserta o actualiza vectores |
| Buscar similares | `index.query()` | Búsqueda semántica por similitud |
| Buscar por ID | `index.fetch()` | Búsqueda exacta por ID |
| Actualizar metadata | `index.update()` | Modifica metadata de un vector |
| Eliminar | `index.delete()` | Por ID, filtro o todos |
| Listar IDs | `index.list_paginated()` | Lista todos los IDs del index |
| Estadísticas | `index.describe_index_stats()` | Total vectores, namespaces, etc. |

---

## 📝 Ejemplos paso a paso

### 01 — Conexión básica

```python
from pinecone import Pinecone

pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
indexes = pc.list_indexes()
print(indexes.names())  # []
```

### 02 — Crear un index

```python
pc.create_index(
    name="mi-index",
    dimension=768,
    metric="cosine",
    spec=ServerlessSpec(cloud="aws", region="us-east-1")
)
```

### 03 — Upsert de vectores

```python
index.upsert(vectors=[
    {
        "id": "doc_1",
        "values": [0.21, -0.87, ...],        # vector de 768 números
        "metadata": {
            "texto": "El gato duerme",
            "categoria": "animales"
        }
    }
])
```

### 04 — Query (búsqueda semántica)

```python
resultados = index.query(
    vector=embedding_pregunta,
    top_k=3,                  # máximo de resultados
    include_metadata=True,
    filter={"categoria": {"$eq": "animales"}}  # opcional
)

for match in resultados.matches:
    print(match.id, match.score, match.metadata)
```

> `top_k` es un **máximo**, no un mínimo. Si hay menos vectores que `top_k`, devuelve los que existan.

### 05 — Eliminar vectores

```python
index.delete(ids=["doc_1"])                               # por ID
index.delete(ids=["doc_1", "doc_2"])                      # varios
index.delete(filter={"categoria": {"$eq": "animales"}})   # por filtro
index.delete(delete_all=True)                             # todos
```

> ⚠️ Pinecone es **eventual consistency** — después de un delete espera 1-2 segundos antes de consultar.

### 06 — Fetch, Update y List

```python
# Fetch por ID — búsqueda exacta, devuelve values + metadata
resultado = index.fetch(ids=["doc_1"])
vector = resultado.vectors["doc_1"]
print(vector.metadata)

# Update — solo modifica metadata, no necesitas el vector completo
index.update(
    id="doc_1",
    set_metadata={"texto": "nuevo texto", "actualizado": True}
)

# List — devuelve solo IDs, no values ni metadata
resultado_ids = index.list_paginated()
for v in resultado_ids.vectors:
    print(v.id)
```

### Filtros de metadata

Pinecone usa un lenguaje de filtros inspirado en MongoDB que aplica **solo sobre la metadata**:

| Operador | Significado | Ejemplo |
|----------|-------------|---------|
| `$eq` | igual a | `{"categoria": {"$eq": "animales"}}` |
| `$ne` | no igual a | `{"categoria": {"$ne": "animales"}}` |
| `$gt` | mayor que | `{"precio": {"$gt": 100}}` |
| `$gte` | mayor o igual | `{"precio": {"$gte": 100}}` |
| `$lt` | menor que | `{"precio": {"$lt": 100}}` |
| `$lte` | menor o igual | `{"precio": {"$lte": 100}}` |
| `$in` | está en lista | `{"categoria": {"$in": ["animales", "comida"]}}` |
| `$nin` | no está en lista | `{"categoria": {"$nin": ["tecnologia"]}}` |

---

## 🤖 RAG Completo — Ejemplo 07

RAG = **Retrieval-Augmented Generation**

En lugar de que el LLM responda con su conocimiento general, primero buscamos los fragmentos más relevantes del documento y se los damos como contexto. El LLM solo responde basándose en ese contexto.

### Flujo completo

```
PDF (guia_nutricion.pdf)
        ↓ pymupdf extrae el texto página por página
Chunks de texto (fragmentos de 800 caracteres con 150 de overlap)
        ↓ sentence-transformers genera embeddings
Vectores de 768 dimensiones + metadata (página, documento)
        ↓ upsert
Pinecone Index "rag-nutricion"
        ↑ query top-5 más similares
Pregunta del usuario → convertida a embedding
        ↓ contexto (chunks relevantes) + pregunta
Ollama (qwen2.5:7b) corriendo localmente
        ↓
Respuesta basada ÚNICAMENTE en el PDF
```

### ¿Qué es un chunk y el overlap?

```
Texto: "...proteínas son esenciales para construir tejidos..."
chunk_size=800, overlap=150

Chunk 1: caracteres 0    → 800
Chunk 2: caracteres 650  → 1450   ← comparte 150 chars con chunk 1
Chunk 3: caracteres 1300 → 2100   ← comparte 150 chars con chunk 2
```

El overlap evita que información importante quede cortada entre dos chunks.

### Metadata guardada por chunk

```python
{
    "id": "chunk_5",
    "values": [0.21, -0.87, ...],   # 768 números
    "metadata": {
        "texto": "...fragmento del PDF...",
        "pagina": 3,                 # página del PDF
        "documento": "guia_nutricion.pdf",
        "chunk_id": 5
    }
}
```

### Resultados de prueba

| Pregunta | Score más alto | Página | Respuesta correcta |
|----------|---------------|--------|--------------------|
| ¿Cuánta proteína necesito al día? | 0.690 | Pág. 3 | ✅ 0.8g/kg sedentarios, 1.2-2.0g/kg activos |
| ¿Qué alimentos tienen más calcio? | 0.599 | Pág. 8 | ✅ Lácteos, sardinas, brócoli, almendras |
| ¿Es malo comer carbohidratos? | 0.686 | Pág. 3 y 8 | ✅ No, son la principal fuente de energía |
| ¿Mejores ejercicios para bajar de peso? | 0.534 | — | ✅ "El contexto no contiene esa información" |

> La última respuesta es clave — el sistema **no inventa**. Si la información no está en el PDF, lo dice claramente.

---

## 🔄 Cambiar el LLM

Actualmente el proyecto usa **Ollama con qwen2.5:7b** de forma local y gratuita. Puedes cambiarlo fácilmente:

### Opción A — Gemini (Google, gratuito)

```bash
pip install google-generativeai
```

```python
# En el .env agrega:
# GOOGLE_API_KEY=AIzaxxxxxxxxxx

import google.generativeai as genai
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
modelo = genai.GenerativeModel("gemini-2.0-flash")

respuesta = modelo.generate_content(prompt)
print(respuesta.text)
```

### Opción B — Ollama local (actual, gratuito)

```python
import ollama

respuesta = ollama.chat(
    model="qwen2.5:7b",
    messages=[{"role": "user", "content": prompt}]
)
print(respuesta['message']['content'])
```

Otros modelos disponibles en Ollama:
```bash
ollama pull llama3.2:3b    # más ligero
ollama pull mistral:7b     # buena calidad
ollama pull qwen2.5:7b     # el que usamos (bueno en español con prompt)
```

### Opción C — OpenAI

```bash
pip install openai
```

```python
# En el .env agrega:
# OPENAI_API_KEY=sk-xxxxxxxxxx

from openai import OpenAI
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

respuesta = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": prompt}]
)
print(respuesta.choices[0].message.content)
```

---

## 🔑 Notas técnicas importantes

### load_dotenv vs os.getenv

```python
load_dotenv()        # ESCRIBE las variables en os.environ (deposita)
os.getenv("KEY")     # LEE las variables de os.environ (recoge)
```

Son dos librerías distintas que trabajan juntas. Sin `load_dotenv()` primero, `os.getenv()` devuelve `None`.

### Path(__file__) para encontrar el .env

```python
# Siempre encuentra el .env sin importar desde dónde corras el script
load_dotenv(Path(__file__).parent.parent / ".env")

# Path(__file__)                → /ruta/proyecto/ejemplos/07_rag_completo.py
# Path(__file__).parent         → /ruta/proyecto/ejemplos/
# Path(__file__).parent.parent  → /ruta/proyecto/
# + ".env"                      → /ruta/proyecto/.env
```

### fetch vs query

| Método | ¿Cómo busca? | Devuelve |
|--------|-------------|----------|
| `fetch` | Por ID exacto | values + metadata |
| `query` | Por similitud semántica | score + metadata |
| `list_paginated` | Lista todos | solo IDs |

---

## 🚀 Tecnologías usadas

| Tecnología | Versión | Uso |
|------------|---------|-----|
| [Pinecone](https://www.pinecone.io) | 9.0.1 | Base de datos vectorial |
| [sentence-transformers](https://www.sbert.net) | 5.5.0 | Generar embeddings multilingües |
| [pymupdf](https://pymupdf.readthedocs.io) | 1.27.2 | Extraer texto del PDF |
| [Ollama](https://ollama.com) | — | LLM local (qwen2.5:7b) |
| [python-dotenv](https://github.com/theskumar/python-dotenv) | — | Manejo de variables de entorno |

---

*Workshop creado para aprender Pinecone desde cero — desde conexión básica hasta RAG con PDF real en español.*
