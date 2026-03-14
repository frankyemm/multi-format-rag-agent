# Multi-Format RAG Agent - Analista de Datos IA

## Descripción de la Solución

Este proyecto implementa un **Agente de Análisis de Datos IA con RAG (Retrieval-Augmented Generation)** diseñado para procesar documentos en múltiples formatos y responder preguntas basadas exclusivamente en su contenido, siguiendo un protocolo de "no-hallucination" (sin invenciones). 

El agente permite:
- **Subida de documentos**: Soporte para PDF, DOCX, XLSX y TXT.
- **Procesamiento inteligente**: Extracción de texto, chunking semántico adaptativo, embeddings locales y almacenamiento vectorial.
- **Consultas en lenguaje natural**: Respuestas generadas por LLM (OpenAI GPT-4o-mini) basadas en contexto relevante de los documentos.
- **Interfaz REST API**: Endpoints para gestión de documentos y consultas.

Es ideal para análisis de datos estructurados y no estructurados, asegurando respuestas precisas y citadas con fuentes.

## Arquitectura Utilizada

El sistema sigue una arquitectura modular y stateless:

- **Backend (FastAPI)**: API REST para endpoints de subida, consulta y gestión de documentos.
- **Carga de Documentos**: Módulo `document_loader.py` con loaders especializados para cada formato (PyMuPDF para PDF, python-docx para DOCX, pandas para XLSX, etc.).
- **Chunking Semántico**: `rag_engine.py` implementa chunking adaptativo basado en similitud semántica entre oraciones para mejorar la recuperación.
- **Embeddings**: Modelo `sentence-transformers/all-MiniLM-L6-v2` (384-dimensiones) ejecutado en CPU para compatibilidad.
- **Base de Datos Vectorial**: ChromaDB persistente para almacenamiento y búsqueda eficiente de embeddings.
- **Generación de Respuestas**: OpenAI GPT-4o-mini con prompts construidos dinámicamente para incluir contexto y fuentes.
- **Configuración**: Variables de entorno (.env) y módulo `config.py` para parámetros ajustables.
- **Frontend**: Archivos estáticos servidos por FastAPI (index.html en `/static`).

Diagrama simplificado:
```
Usuario → FastAPI API → Document Loader → RAG Engine (Chunking + Embeddings + Retrieval) → OpenAI LLM → Respuesta
```

## Instrucciones para Ejecutar el Proyecto

### Prerrequisitos
- Python 3.8+
- Clave de API de OpenAI (obtén una en [OpenAI Platform](https://platform.openai.com/api-keys))

### Instalación
1. Clona o descarga el repositorio.
2. Crea un entorno virtual:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # En Windows: .venv\Scripts\activate
   ```
3. Instala dependencias:
   ```bash
   pip install -r requirements.txt
   ```
4. Configura variables de entorno: Crea un archivo `.env` en la raíz con:
   ```
   OPENAI_API_KEY=tu_clave_aqui
   ```

### Ejecución
1. Inicia el servidor:
   ```bash
   uvicorn app.main:app --reload
   ```
2. Abre en navegador: `http://localhost:8000`

### Uso
- **Subir documentos**: Usa el endpoint `POST /upload` (o la interfaz web).
- **Hacer preguntas**: `POST /ask` con JSON `{"question": "Tu pregunta"}`.
- **Listar documentos**: `GET /documents`.
- **Eliminar documentos**: `DELETE /documents/{doc_id}`.

### Detener
Presiona `Ctrl+C` en la terminal.

## Decisiones Técnicas Tomadas

- **Embeddings en CPU**: Forzado `device="cpu"` para evitar errores de CUDA en GPUs incompatibles, priorizando estabilidad sobre velocidad.
- **Chunking Semántico Adaptativo**: En lugar de chunks fijos por caracteres, se usa similitud coseno entre embeddings de oraciones para agrupar contenido relacionado, mejorando la calidad de recuperación sin perder contexto.
- **Top-K Retrieval Aumentado**: Se recupera hasta 8 chunks relevantes (vs. estándar 5) para mayor cobertura en consultas complejas.
- **Manejo de Errores en HNSW**: Fallback gracioso para colecciones pequeñas en ChromaDB, evitando crashes por parámetros ef/M demasiado bajos.
- **Protocolo No-Hallucination**: Respuestas basadas únicamente en documentos cargados; el prompt incluye fuentes explícitas y advierte al LLM de no inventar información.
- **Persistencia Simple**: Registro de documentos en JSON plano para simplicidad; ChromaDB para vectores.
- **Idioma y Formato**: Todo en español, con tablas Markdown para datos tabulares (XLSX) para preservar estructura en RAG.
- **Dependencias Minimalistas**: Elegidas por estabilidad y compatibilidad (e.g., ChromaDB 0.6.3, no la última versión para evitar breaking changes).

## Posibles Mejoras Futuras

- **Memoria de Conversación**: Agregar historial de mensajes para consultas contextuales (e.g., "según mi pregunta anterior...").
- **Sistema de Feedback**: Implementar like/dislike en respuestas para recopilar datos y ajustar prompts/calidad dinámicamente.
- **Soporte Adicional de Formatos**: Imágenes (OCR), CSV, JSON, o bases de datos.
- **Fine-Tuning del Modelo**: Entrenar un modelo personalizado con datos específicos del dominio para reducir dependencia de OpenAI.
- **Interfaz de Usuario Mejorada**: SPA completa con React/Vue para subir archivos y chat en tiempo real.
- **Caching y Optimización**: Cache de embeddings para documentos recurrentes; búsqueda híbrida (BM25 + vectorial).
- **Autenticación y Multi-Usuario**: Soporte para usuarios separados con sus propios documentos/vectores.
- **Escalabilidad**: Migrar a vector DBs como Pinecone o Weaviate para grandes volúmenes; contenedorización con Docker.
- **Monitoreo y Logging**: Agregar métricas de uso, logs detallados y alertas para errores.
- **Multilingüe**: Soporte para documentos en múltiples idiomas con modelos de embeddings apropiados.
- **Integración con Herramientas Externas**: Conexión a APIs de datos (e.g., Google Sheets) o herramientas de BI.

---

**Autor**: Franky Cardona, Cascade AI Assistant.
**Licencia**: MIT (asumida, ajusta según necesites).
