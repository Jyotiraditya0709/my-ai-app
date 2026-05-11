# FastAPI in Production

FastAPI is a modern Python web framework designed for building APIs. It supports async/await natively and uses Pydantic for validation. This makes it well-suited for AI applications where LLM calls are slow.

## Async fundamentals

The async model is built on Python's asyncio. Every endpoint defined with async def runs on the event loop. This means a single worker can handle many concurrent requests, since each one yields control during I/O operations like LLM calls or database queries.

When a request hits the server, FastAPI dispatches it to the right handler based on the route. The handler runs to its first await, then yields. While it's yielded, other requests can run.

## Pydantic validation

Pydantic models define the shape of request and response data. FastAPI uses these models for two things automatically: validation of incoming data, and generation of the OpenAPI schema for docs.

You define a model by inheriting from BaseModel and adding fields with type hints. Pydantic enforces types at runtime, returning a 422 error if the request body doesn't match.

## Dependency injection

FastAPI's Depends() system injects values into your endpoint functions. Common uses: database connections, the current authenticated user, the request settings. Dependencies can be sync or async, and they chain — a dependency can itself depend on other dependencies.

This pattern keeps endpoint code clean. Authentication, DB pooling, and config loading happen in dependencies, not in every endpoint.