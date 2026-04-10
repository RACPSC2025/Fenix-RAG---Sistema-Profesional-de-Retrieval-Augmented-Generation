# 🏗️ Design Patterns en Arquitecturas Reales

> Singleton, Factory Method, Debounce aplicados en código de producción.
> No por nombre — por problema real que resuelven.

---

## 1. Singleton

### Problema Real que Resuelve
"Necesito UNA sola instancia de algo costoso de crear, compartida en toda la app."

### Mal Ejemplo (Anti-Pattern)
```python
# ❌ Singleton con variable global
_llm_client = None

def get_llm():
    global _llm_client
    if _llm_client is None:
        _llm_client = ChatBedrock(model="amazon.nova-pro-v1:0")
    return _llm_client
```

### Buen Ejemplo (Thread-Safe + Async-Safe)
```python
from functools import lru_cache
from langchain_aws import ChatBedrock

@lru_cache(maxsize=4)  # Cache por modelo
def get_llm(model: str = "amazon.nova-pro-v1:0", temperature: float = 0.0):
    """
    Singleton con cache por parametros.
    
    - Thread-safe (lru_cache es thread-safe en CPython)
    - Async-safe (no muta estado compartido)
    - Per multiples modelos (no solo uno)
    """
    return ChatBedrock(model=model, temperature=temperature)

# Uso
llm_pro = get_llm("amazon.nova-pro-v1:0")
llm_lite = get_llm("amazon.nova-lite-v1:0")
# Ambas comparten instancias cacheadas
```

### Dónde lo Usamos en RACodex
- `get_settings()` — Settings con `@lru_cache`
- `get_llm()` — LLM clients con `@lru_cache`
- `get_vector_store()` — Chroma connections con singleton

### Cuándo NO Usar Singleton
- Cuando necesitas mockear en tests (usa dependency injection)
- Cuando el objeto tiene estado mutable compartido (race conditions)
- Cuando necesitas multiples instancias por tenant (multi-tenancy)

---

## 2. Factory Method

### Problema Real que Resuelve
"Necesito crear diferentes tipos de objetos sin saber cual hasta runtime."

### Mal Ejemplo (Anti-Pattern)
```python
# ❌ if/elif infinito — viola Open/Closed
def get_loader(file_type: str):
    if file_type == "pdf":
        return PyMuPDFLoader()
    elif file_type == "word":
        return WordLoader()
    elif file_type == "excel":
        return ExcelLoader()
    elif file_type == "image":
        return OCRLoader()
    # Cada nuevo tipo requiere modificar esta funcion
    else:
        raise ValueError(f"Unknown type: {file_type}")
```

### Buen Ejemplo (Factory Method con Registry)
```python
from typing import Protocol, Type

class DocumentLoader(Protocol):
    def load(self, file_path: str) -> list[Document]: ...

class LoaderRegistry:
    _loaders: dict[str, Type[DocumentLoader]] = {}
    
    @classmethod
    def register(cls, file_type: str):
        """Decorator para registrar loaders."""
        def decorator(loader_cls):
            cls._loaders[file_type] = loader_cls
            return loader_cls
        return decorator
    
    @classmethod
    def get_loader(cls, file_type: str) -> DocumentLoader:
        if file_type not in cls._loaders:
            raise ValueError(f"No loader for: {file_type}")
        return cls._loaders[file_type]()

# Registrar loaders
@LoaderRegistry.register("pdf")
class PyMuPDFLoader:
    def load(self, file_path: str) -> list[Document]: ...

@LoaderRegistry.register("word")
class WordLoader:
    def load(self, file_path: str) -> list[Document]: ...

# Uso — agregar un nuevo loader NO requiere modificar el factory
loader = LoaderRegistry.get_loader("pdf")
docs = loader.load("document.pdf")
```

### Dónde lo Usamos en RACodex
- `LoaderRegistry` en `src/ingestion/registry.py` — selecciona loader por MIME type
- `get_ensemble_retriever()` — factory para retrievers con diferentes estrategias

### Pregunta de Mentoría
**Junior:** "¿Por qué es mejor el registry que el if/elif?"  
**Mentor:** "Porque si mañana necesitas agregar soporte para `.pptx`, con el registry solo creas la clase nueva y la decoras. Con el if/elif, modificas codigo que ya funciona (y puedes romperlo). Eso se llama Open/Closed Principle."

---

## 3. Debounce

### Problema Real que Resuelve
"Evitar ejecutar algo costoso muchas veces en poco tiempo."

### Ejemplo Real: Control de Consumo de Tokens
```python
import time
from functools import wraps
from collections import defaultdict

class TokenDebounce:
    """
    Evita que un usuario haga mas de N requests de LLM en M segundos.
    
    No es rate limiting HTTP — es control a nivel de aplicacion
    para proteger el presupuesto de tokens.
    """
    def __init__(self, max_calls: int = 10, window_seconds: float = 60.0):
        self.max_calls = max_calls
        self.window = window_seconds
        self._calls: dict[str, list[float]] = defaultdict(list)
    
    def __call__(self, func):
        @wraps(func)
        def wrapper(user_id: str, *args, **kwargs):
            now = time.time()
            # Limpiar llamadas fuera de la ventana
            self._calls[user_id] = [
                t for t in self._calls[user_id] if now - t < self.window
            ]
            
            if len(self._calls[user_id]) >= self.max_calls:
                raise TokenLimitExceeded(
                    f"User {user_id} exceeded {self.max_calls} calls in {self.window}s"
                )
            
            self._calls[user_id].append(now)
            return func(user_id, *args, **kwargs)
        return wrapper

# Uso
debounce = TokenDebounce(max_calls=10, window_seconds=60)

@debounce
def call_llm_for_user(user_id: str, query: str):
    llm = get_llm()
    return llm.invoke(query)

# Si un usuario hace 11 llamadas en 60s → excepcion
```

### Debounce en Frontend (Búsqueda con RAG)
```javascript
// React — debounce en search input
import { useState, useCallback } from 'react';
import debounce from 'lodash/debounce';

function SearchWithRAG() {
  const [query, setQuery] = useState('');
  
  // Debounce de 500ms — no llama a la API en cada keystroke
  const debouncedSearch = useCallback(
    debounce((q) => fetch(`/api/search?q=${q}`), 500),
    []
  );
  
  return (
    <input 
      value={query}
      onChange={(e) => {
        setQuery(e.target.value);
        debouncedSearch(e.target.value);
      }}
    />
  );
}
```

---

## 4. Strategy Pattern

### Problema Real
"Necesito intercambiar algoritmos en runtime sin cambiar el codigo que los usa."

### Ejemplo Real: Retrieval Strategies
```python
from abc import ABC, abstractmethod

class RetrievalStrategy(ABC):
    @abstractmethod
    def retrieve(self, query: str, top_k: int) -> list[Document]: ...

class HybridStrategy(RetrievalStrategy):
    def retrieve(self, query: str, top_k: int) -> list[Document]:
        # BM25 + Vector con RRF
        return hybrid_retrieve(query, top_k)

class ParentChildStrategy(RetrievalStrategy):
    def retrieve(self, query: str, top_k: int) -> list[Document]:
        # Busca en hijos, retorna padres
        return parent_child_retrieve(query, top_k)

class SemanticOnlyStrategy(RetrievalStrategy):
    def retrieve(self, query: str, top_k: int) -> list[Document]:
        # Solo embeddings semanticos
        return semantic_retrieve(query, top_k)

# Contexto que usa la estrategia
class Retriever:
    def __init__(self, strategy: RetrievalStrategy):
        self.strategy = strategy
    
    def query(self, text: str, k: int = 5) -> list[Document]:
        return self.strategy.retrieve(text, k)

# Cambio de estrategia en runtime
retriever = Retriever(HybridStrategy())  # Default
retriever.strategy = ParentChildStrategy()  # Cambiar sin tocar el Retriever
```

---

## 5. Observer Pattern

### Problema Real
"Cuando algo cambia, necesito notificar a multiples interesados sin acoplarlos."

### Ejemplo Real: Event-Driven Ingestion
```python
from typing import Callable, Protocol

class IngestionEvent(Protocol):
    document_id: str
    status: str  # "started", "completed", "failed"
    error: str | None

class IngestionObserver:
    """
    Observer para eventos de ingestion.
    Multiple handlers se suscriben sin conocerse entre si.
    """
    def __init__(self):
        self._handlers: list[Callable[[IngestionEvent], None]] = []
    
    def subscribe(self, handler: Callable[[IngestionEvent], None]):
        self._handlers.append(handler)
    
    def notify(self, event: IngestionEvent):
        for handler in self._handlers:
            try:
                handler(event)
            except Exception as e:
                # Un handler no debe romper a los demas
                log.error("Observer handler failed", error=str(e))

# Uso
observer = IngestionObserver()

# Handler 1: Logging
observer.subscribe(lambda e: log.info(f"Ingestion {e.status}: {e.document_id}"))

# Handler 2: Metricas
observer.subscribe(lambda e: metrics.increment(f"ingestion.{e.status}"))

# Handler 3: Notificacion al usuario
observer.subscribe(lambda e: send_notification(e.document_id, e.status))

# Disparar
event = IngestionEvent(document_id="doc-123", status="completed", error=None)
observer.notify(event)  # Los 3 handlers se ejecutan
```

---

## Checklist de Seniority

| Pattern | Junior | Mid | Senior |
|---------|--------|-----|--------|
| Singleton | Lo implementa con variable global | Usa `@lru_cache` | Sabe cuando NO usarlo |
| Factory Method | if/elif infinito | Registry pattern | Factory + Strategy combinados |
| Debounce | No lo conoce | Lo aplica en frontend | Lo aplica en backend para costos |
| Strategy | No lo conoce | Lo usa con retrievers | Lo usa con LLM providers + retrieval + chunking |
| Observer | Signals de Django | Event bus explicito | Observer + async + error isolation |

---

*Referencia: Refactoring Guru — https://refactoring.guru/design-patterns*
