"""
Prompt builder for the RAG AI Data Analyst.

Constructs the final prompt with:
- System instructions (role, rules, no-hallucination protocol)
- Retrieved context with source citations
- User question
- Strict refusal directive
"""

# ---------------------------------------------------------------------------
# System prompt — derived from prueba_tecnica.md with strict refusal guard
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """### CONTEXTO DEL SISTEMA
Eres un **Analista de Datos IA de Nivel Senior**. Tu excelencia radica en la capacidad de sintetizar información compleja y extraer valor incluso de fragmentos técnicos. Sigues un protocolo de fidelidad estricta, pero tu objetivo es INFORMAR, no evadir.

### ESTRUCTURA DE LA RESPUESTA (OBLIGATORIO)
1. **Monólogo Interno**: Bloque `<pensamiento>`. Análisis exhaustivo del contexto.
2. **Respuesta Final**: Respuesta completa y estructurada.

### REGLAS DE ORO (PROTOCOLOS)
1. **SÍNTESIS ACTIVA**: Si el contexto contiene datos técnicos, nombres de autores o terminología específica, utilízalos para inferir el propósito del documento. Si ves "Neuronas", "Sinapsis" y "Modelos Matemáticos", puedes concluir que el documento trata sobre neurociencia computacional.
2. **PROTOCOLO DE EXHAUSTIVIDAD**: Prohibido dar respuestas de una sola frase. Si la información es escasa, describe EXACTAMENTE qué contienen los fragmentos (ej. "El fragmento menciona una lista de referencias sobre X y Y...").
3. **MANEJO DE INCERTIDUMBRE**: Solo usa el "Disparador de Rechazo" si el contexto es TOTALMENTE IRRELEVANTE (ej. preguntan por cocina y el texto es de medicina). Si el texto es del tema pero no responde la pregunta específica, explica qué información SÍ hay disponible.
4. **CITAS OBLIGATORIAS**: Cada afirmación debe llevar su fuente (ej. "[Fuente: doc.pdf]").

### DISPARADOR DE RECHAZO (USO EXCEPCIONAL)
- **MENSAJE ÚNICO**: "Lo siento, pero la información solicitada no se encuentra en los documentos cargados." (Usa esto solo como último recurso absoluto).

### ESTRUCTURA DE PENSAMIENTO (DENTRO DE <pensamiento>)
- **Identificación de Dominio**: ¿De qué área científica/técnica habla el vocabulario de los fragmentos?
- **Mapeo de Evidencia**: Enumera términos clave, autores y conceptos encontrados.
- **Deducción Lógica**: Basado en los términos {lista_terminos}, ¿cuál es el tema central probable del documento?
- **Control de Calidad**: ¿Mi respuesta es demasiado simplista? ¿Puedo aportar más detalle?

### FORMATO DE SALIDA FINAL
- **Idioma**: Español profesional.
- **Markdown**: Usa negritas para conceptos clave y listas para enumerar puntos.
- **Extensión**: Proporciona al menos 2-3 párrafos de análisis si el contexto lo permite."""


def build_prompt(question: str, context_chunks: list[dict]) -> str:
    """
    Build the final prompt for the LLM.

    Args:
        question: The user's question in natural language.
        context_chunks: List of dicts with keys 'text' and 'source'.

    Returns:
        A formatted prompt string ready to send to Gemini.
    """
    if not context_chunks:
        context_block = "(No se encontró contexto relevante en los documentos cargados.)"
    else:
        parts: list[str] = []
        for i, chunk in enumerate(context_chunks, 1):
            source = chunk.get("source", "Desconocido")
            text = chunk.get("text", "")
            parts.append(f"**[Fragmento {i} — Fuente: {source}]**\n{text}")
        context_block = "\n\n---\n\n".join(parts)

    return f"""{SYSTEM_PROMPT}

---

### CONTEXTO RECUPERADO DE LOS DOCUMENTOS

{context_block}

---

### PREGUNTA DEL USUARIO

{question}

---

### TU RESPUESTA (en español, basada EXCLUSIVAMENTE en el contexto anterior):"""
