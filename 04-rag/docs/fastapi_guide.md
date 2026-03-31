# FastAPI Production Guide

## Getting Started

FastAPI is a modern Python web framework built on top of Starlette and Pydantic. It provides automatic OpenAPI documentation, request validation, and async support out of the box.

To install FastAPI, run: `pip install fastapi uvicorn`

## Project Structure

A well-organized FastAPI project should follow this structure:
- `main.py` — Application entry point and route registration
- `routers/` — Route modules grouped by domain
- `models/` — Pydantic models for request/response validation
- `services/` — Business logic layer
- `repositories/` — Data access layer

## Dependency Injection

FastAPI's dependency injection system uses `Depends()`. Define dependencies as functions and inject them into route handlers. This is the recommended way to manage database sessions, authentication, and shared services.

```python
from fastapi import Depends

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@app.get("/items")
def read_items(db: Session = Depends(get_db)):
    return db.query(Item).all()
```

## Authentication

For production APIs, use OAuth2 with JWT tokens. FastAPI provides `OAuth2PasswordBearer` for token extraction. Always validate tokens on every request using a dependency.

## Error Handling

Use HTTPException for expected errors and custom exception handlers for unexpected ones. Always return consistent error response shapes.

## Performance Tips

- Use async endpoints for I/O-bound operations
- Connection pooling is critical — configure SQLAlchemy's pool_size and max_overflow
- Add response caching with Redis for expensive queries
- Use background tasks for non-blocking operations like sending emails
- Profile with `py-spy` before optimizing

## Deployment

Deploy with Gunicorn + Uvicorn workers in production:
```
gunicorn main:app -w 4 -k uvicorn.workers.UvicornWorker
```

Use at least 2 * CPU cores + 1 workers. Put Nginx or a cloud load balancer in front for TLS termination and rate limiting.
