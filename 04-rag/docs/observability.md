# Observability for Production Services

## The Three Pillars

Observability is built on three pillars: logs, metrics, and traces. Each serves a different purpose and they complement each other.

### Structured Logging

Use structured (JSON) logging in production. Unstructured log messages are nearly impossible to query at scale.

```python
import structlog

logger = structlog.get_logger()

# Bad: unstructured
logger.info("User logged in from 192.168.1.1")

# Good: structured
logger.info("user_login", user_id="u_123", ip="192.168.1.1", method="oauth2")
```

Ship logs to a centralized system like Elasticsearch, Loki, or Datadog. Set log levels appropriately: DEBUG for development, INFO for production, ERROR for alerts.

### Metrics

Use Prometheus-style metrics for real-time monitoring. The four golden signals to track:
- Latency: How long requests take (use histograms, not averages)
- Traffic: Requests per second
- Errors: Error rate as a percentage of total requests
- Saturation: How close to capacity your service is

Export metrics via a `/metrics` endpoint and scrape with Prometheus. Use Grafana for dashboards.

### Distributed Tracing

When a request spans multiple services, tracing shows the full journey. Use OpenTelemetry for instrumentation. Each request gets a trace ID that propagates across service boundaries.

Key concepts:
- Trace: The complete lifecycle of a request
- Span: A single operation within a trace
- Context propagation: Passing trace IDs between services via headers

## Alerting Best Practices

Alert on symptoms, not causes. Alert on "error rate > 1%" rather than "CPU > 80%". Keep alerts actionable — every alert should require human intervention. Use PagerDuty or Opsgenie for on-call routing.

## Health Checks

Implement both liveness and readiness probes:
- Liveness: "Is the process alive?" — restart if not
- Readiness: "Can it serve traffic?" — remove from load balancer if not

A readiness check should verify database connectivity, cache availability, and any critical downstream dependencies.
