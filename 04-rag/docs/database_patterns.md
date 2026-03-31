# Database Patterns for Backend Services

## Connection Management

Always use connection pooling in production. Creating a new database connection per request adds 20-50ms of latency. SQLAlchemy's engine manages a pool by default.

Recommended pool settings:
- pool_size: 5-20 (depends on workload)
- max_overflow: 10 (temporary extra connections under load)
- pool_timeout: 30 seconds
- pool_recycle: 1800 seconds (avoid stale connections)

## Migration Strategy

Use Alembic for database migrations. Every schema change should be a versioned migration file. Never modify production databases manually.

Key commands:
- `alembic revision --autogenerate -m "add users table"` — Generate migration
- `alembic upgrade head` — Apply all pending migrations
- `alembic downgrade -1` — Rollback last migration

## Query Optimization

### N+1 Problem
The most common performance issue. When you load a list of items and then load related items one by one, you generate N+1 queries instead of 1-2.

Fix with eager loading:
```python
# Bad: N+1 queries
users = db.query(User).all()
for user in users:
    print(user.orders)  # Each access = 1 query

# Good: 1 query with join
users = db.query(User).options(joinedload(User.orders)).all()
```

### Indexing
Add indexes on columns used in WHERE, JOIN, and ORDER BY clauses. A missing index on a frequently queried column can make queries 100x slower.

## Transaction Patterns

Use the Unit of Work pattern. Open a transaction, perform all operations, commit once. If any operation fails, the entire transaction rolls back.

For distributed systems, consider the Saga pattern instead of distributed transactions. Each service performs its local transaction and publishes events.

## Read Replicas

For read-heavy workloads, route SELECT queries to read replicas and writes to the primary. Implement this at the application level using SQLAlchemy's routing capability.

Be aware of replication lag — after a write, immediately reading from a replica may return stale data. Route post-write reads to the primary for a short window.
