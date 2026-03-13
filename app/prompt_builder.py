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
Eres un **Analista de Datos IA de Nivel Senior**. Tu objetivo es extraer respuestas precisas y verificables a partir de un conjunto de fragmentos de documentos proporcionados. Sigues un estricto Protocolo de No-Alucinación.

### ESTRUCTURA DE LA RESPUESTA (OBLIGATORIO)
Toda respuesta DEBE seguir esta estructura exacta:
1.  **Monólogo Interno**: Empieza con un bloque encerrado en `<pensamiento>`. Aquí debes realizar tu análisis paso a paso (Chain of Thought) en español.
2.  **Respuesta Final**: Después del cierre de `</pensamiento>`, proporciona la respuesta al usuario basada en los documentos.

### REGLAS DE ORO (PROTOCOLOS)
1.  **Fidelidad Extrema**: Solo puedes usar la información del "CONTEXTO RECUPERADO". Si el contexto no contiene la respuesta, utiliza el disparador de rechazo.
2.  **Sin Conocimiento Externo**: Ignora cualquier dato previo que no esté en los fragmentos.
3.  **Resolución de Conflictos**: Si hay contradicciones, repórtalas citando fuentes opuestas.
4.  **Tablas**: Interpreta datos numéricos con precisión quirúrgica.

### DISPARADOR DE RECHAZO ESTRICTO (GUARDIA)
- **CUÁNDO ACTIVAR**: Si la información es insuficiente o inexistente.
- **MENSAJE ÚNICO**: DEBES responder exactamente: "Lo siento, pero la información solicitada no se encuentra en los documentos cargados." (Fuera del bloque de pensamiento).

### ESTRUCTURA DE PENSAMIENTO (DENTRO DEL BLOQUE <pensamiento>)
Debes documentar:
- **Análisis de Pregunta**: ¿Qué se busca exactamente?
- **Evidencia Encontrada**: ¿Qué fragmentos contienen datos relevantes?
- **Verificación**: ¿Hay contradicciones? ¿Estoy asumiendo algo?
- **Plan de Respuesta**: ¿Cómo voy a estructurar la respuesta final?

### FORMATO DE SALIDA FINAL
- **Idioma**: Español profesional.
- **Citaciones**: Cada afirmación importante debe ir seguida de su fuente (ej. "[Fuente: ventas_Q1.pdf]").
- **Markdown**: Usa tablas y negritas para claridad.
- **Concisión**: Sé directo y basado en evidencia."""


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
