# 🐍 Python Async en Django

> asgiref, asyncio, context vars, async tasks, crons, signals y threading.
> Con enfoque dual: explicar a juniors + implementar para seniors.

---

## 1. `async def` en Django Views

### Concepto Básico
```python
# Sincrono (tradicional)
def my_view(request):
    data = MyModel.objects.all()
    return render(request, "template.html", {"data": data})

# Asincrono (Django 3.1+)
async def my_view(request):
    data = await MyModel.objects.all()  # Requiere async ORM o sync_to_async
    return render(request, "template.html", {"data": data})
```

### Cuándo Usar Async
- **Hazlo async cuando:** Llamas a APIs externas, LLMs, o servicios con latencia alta
- **No lo hagas async cuando:** Solo haces queries simples al DB (overhead innecesario)

### Ejemplo Real: Vista que Consulta BD + Llama a LLM
```python
from asgiref.sync import sync_to_async
from langchain_aws import ChatBedrock

@sync_to_async
def get_user_context(user_id):
    """Envuelve operacion sync en async."""
    return UserContext.objects.select_related('profile').get(user_id=user_id)

async def chat_view(request):
    # 1. Obtener contexto de usuario (async-safe)
    context = await get_user_context(request.user.id)
    
    # 2. Llamar a LLM (ya es async nativo)
    llm = ChatBedrock(model="amazon.nova-pro-v1:0")
    response = await llm.ainvoke(context.messages)
    
    return JsonResponse({"response": response.content})
```

### ⚠️ Error Común de Junior
```python
# ❌ MAL: Mezclar sync y async sin convertir
async def bad_view(request):
    data = MyModel.objects.all()  # Esto BLOQUEA el event loop
    return JsonResponse({"data": list(data)})

# ✅ BIEN: Usar sync_to_async
@sync_to_async
def get_data():
    return list(MyModel.objects.all())

async def good_view(request):
    data = await get_data()
    return JsonResponse({"data": data})
```

**Pregunta de comprensión:** ¿Por qué `MyModel.objects.all()` bloquea el event loop?  
*Respuesta:* Porque el ORM de Django usa psycopg (sync) internamente. Sin `sync_to_async`, toda la coroutine espera a que la query sync termine, bloqueando otras requests.

---

## 2. asgiref: `sync_to_async` y `async_to_sync`

### `sync_to_async` — De Sync a Async
```python
from asgiref.sync import sync_to_async

# Como funcion
async def my_async_func():
    result = await sync_to_async(sync_function)(arg1, arg2)

# Como decorator
@sync_to_async
def sync_function(arg1, arg2):
    return do_something()
```

### `async_to_sync` — De Async a Sync
```python
from asgiref.sync import async_to_sync

# Para usar en contextos sync (signals, management commands)
def sync_signal_handler(sender, instance, **kwargs):
    result = async_to_sync(async_function)(instance)
```

### Thread Safety
```python
# Por defecto, sync_to_async ejecuta en thread pool
@sync_to_async(thread_sensitive=True)  # Usa el mismo thread (para DB connections)
def db_operation():
    return MyModel.objects.count()
```

**Regla:** Siempre `thread_sensitive=True` cuando toques la base de datos.

---

## 3. Context Vars

### Problema que Resuelven
En async, no hay `request` global como en sync. Context vars permiten pasar datos a través de la call stack sin argumentos explícitos.

```python
from contextvars import ContextVar

# Definir
request_id: ContextVar[str] = ContextVar("request_id", default="unknown")
user_tenant: ContextVar[str] = ContextVar("user_tenant", default="default")

# Set en middleware
async def tenant_middleware(get_response):
    async def middleware(request):
        token = user_tenant.set(request.user.tenant_id)
        try:
            return await get_response(request)
        finally:
            user_tenant.reset(token)
    return middleware

# Get en cualquier lugar del codigo async
def get_current_tenant() -> str:
    return user_tenant.get()
```

### Uso Real: Multi-Tenant con LangGraph
```python
tenant_db: ContextVar[str] = ContextVar("tenant_db")

async def run_agent_for_tenant(tenant_id: str, query: str):
    token = tenant_db.set(f"tenant_{tenant_id}_db")
    try:
        graph = get_graph()
        return await graph.ainvoke({"query": query})
    finally:
        tenant_db.reset(token)
```

---

## 4. Async Tasks con Celery

### Config Basica
```python
# celery.py
from celery import Celery
from asgiref.sync import async_to_sync

app = Celery('myproject')

@app.task
def ingest_document_task(file_path: str):
    """Task que corre en worker separado."""
    # Este es contexto sync — usar async_to_sync si necesitas llamar a async
    from src.agent.graph import run_agent
    result = async_to_sync(run_agent)(
        user_query=f"Ingest {file_path}",
        uploaded_files=[file_path]
    )
    return result
```

### Llamar desde Vista Async
```python
async def upload_view(request):
    # No bloquear la request — enviar a Celery
    ingest_document_task.delay(file_path=str(uploaded_file.path))
    return JsonResponse({"status": "processing", "task_id": task_id})
```

---

## 5. Signals: Sync vs Async

### Django Signals Son Sync por Defecto
```python
from django.db.models.signals import post_save
from django.dispatch import receiver

@receiver(post_save, sender=Document)
def on_document_saved(sender, instance, created, **kwargs):
    if created:
        # ❌ No puedes hacer await aqui
        # Necesitas enviar a task async
        ingest_document_task.delay(str(instance.file_path))
```

### Workaround para Async en Signals
```python
@receiver(post_save, sender=Document)
def on_document_saved(sender, instance, created, **kwargs):
    if created:
        # Opcion 1: Celery task
        ingest_document_task.delay(str(instance.file_path))
        
        # Opcion 2: async_to_sync (si estas en entorno async)
        # async_to_sync(process_document_async)(instance)
```

---

## 6. Threading vs Asyncio

| Aspecto | Threading | Asyncio |
|---------|----------|---------|
| Concurrencia | I/O con GIL release | I/O sin GIL issues |
| CPU-bound | No mejora | No mejora |
| Facilidad | Mas familiar | Curva de aprendizaje |
| Django | Default | Requiere ASGI + config |
| Recomendacion | Legacy code, libs sync | Nuevo codigo, APIs externas |

### Cuándo Usar Cual
```python
# Threading: cuando la libreria no soporta async
import threading
from concurrent.futures import ThreadPoolExecutor

def process_with_threading():
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(sync_function, arg) for arg in args]
        return [f.result() for f in futures]

# Asyncio: cuando controlas todo el stack
async def process_with_asyncio():
    tasks = [async_function(arg) for arg in args]
    return await asyncio.gather(*tasks)
```

---

## Checklist de Seniority

| Tema | Junior | Mid | Senior |
|------|--------|-----|--------|
| `async def` | Sabe escribirlo | Sabe cuando usarlo | Sabe cuando NO usarlo |
| `sync_to_async` | Lo usa siempre | Usa `thread_sensitive` cuando toca | Entiende el thread pool internals |
| Context vars | No los conoce | Los usa para request ID | Los usa para multi-tenant + observabilidad |
| Celery | Envía tasks | Configura queues y retries | Monitorea, escala workers, handlea fallos |
| Signals | Los usa para todo | Los limita a logica simple | Prefiere events explícitos sobre signals |

---

*Referencia: Django Async Docs — https://docs.djangoproject.com/en/stable/topics/async/*
