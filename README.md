RACodex es un agente de desarrollo inteligente que combina **RAG profesional** con **capacidades cognitivas agenticas**. A diferencia de los asistentes de código que dependen exclusivamente del modelo preentrenado, 

RACodex puede:
📚 **Aprender de TUS documentos** — Libros del curso, documentación interna, reglas de la empresa, guías de estilo
🔍 **Responder con fuentes verificables** — "Según Clean Code, capítulo 3, página 42..."
💻 **Codear siguiendo TUS convenciones** — Stack específico, patrones preferidos, políticas de seguridad
🛡️**No alucinar** — Si la respuesta no está en el contexto, lo dice. Sin inventar.

## ¿Para quién?
| Usuario | Problema | Solución RACodex |
|---------|----------|-----------------|
| 🎓 **Estudiante** | El profesor usa un libro que el LLM no conoce | Carga el libro → el agente responde basado en ESE libro |
| 🚀 **Startup** | Convenciones internas que nadie documenta | Ingesta la wiki → el agente codea siguiendo TUS reglas |
| 🏢 **Empresa** | Manuales de compliance, políticas de seguridad | Ingesta los manuales → el agente nunca sugiere algo que viole políticas |
| 💼 **Freelancer** | Stack específico con patrones favoritos | Ingesta tu documentación → el agente replica tu estilo |

## ¿Cómo funciona?
Usuario: "¿Cómo implemento autenticación JWT?" + Carga documentos: [Libro_X, Doc_Interna, Reglas_Empresa]

RACodex:
1. 🧭 Semantic Router → clasifica la query (5ms, 0 tokens)
2. ✍️ Query Transformer → [original, rewritten, step-back]
3. 🔍 Multi-Query Retrieval → busca en TUS documentos con 6 estrategias
4. 🪟 Context Enrichment → agrega contexto de chunks vecinos
5. 📊 CRAG Grade → ¿los docs son relevantes? Si no, reformula y reintenta
6. 🧠 Rethinking Generation → 2 pasadas de lectura para precisión
7. 🔄 Reflection → anti-alucinación: valida la respuesta antes de responder

Resultado: "Según tu libro 'Clean Architecture' (cap. 15, p. 234), el patrón recomendado es Repository + Unit of Work. Aquí está la implementación siguiendo tus reglas de FastAPI..."