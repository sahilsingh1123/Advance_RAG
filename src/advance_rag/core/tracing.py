"""OpenTelemetry configuration for tracing."""

import os
from typing import Dict, Optional

from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.instrumentation.sqlalchemy import SQLAlchemyInstrumentor
from opentelemetry.instrumentation.redis import RedisInstrumentor
from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor
from opentelemetry.propagate import set_global_textmap
from opentelemetry.propagators.b3 import B3MultiFormat

from advance_rag.core.config import get_settings

settings = get_settings()


class TracingConfig:
    """OpenTelemetry tracing configuration."""

    def __init__(self):
        self.enabled = settings.TRACING_ENABLED
        self.sample_rate = settings.TRACING_SAMPLE_RATE
        self.service_name = "advance-rag"
        self.service_version = "0.1.0"
        self.otlp_endpoint = os.getenv(
            "OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4317"
        )
        self.environment = settings.ENVIRONMENT

    def setup_tracing(self, app=None):
        """Setup OpenTelemetry tracing."""
        if not self.enabled:
            return

        # Create resource
        resource = Resource.create(
            attributes={
                "service.name": self.service_name,
                "service.version": self.service_version,
                "deployment.environment": self.environment,
            }
        )

        # Create tracer provider
        tracer_provider = TracerProvider(resource=resource)

        # Create OTLP exporter
        otlp_exporter = OTLPSpanExporter(endpoint=self.otlp_endpoint, insecure=True)

        # Add span processor
        span_processor = BatchSpanProcessor(otlp_exporter)
        tracer_provider.add_span_processor(span_processor)

        # Set global tracer provider
        trace.set_tracer_provider(tracer_provider)

        # Set global propagator
        set_global_textmap(B3MultiFormat())

        # Instrument FastAPI
        if app:
            FastAPIInstrumentor.instrument_app(
                app, tracer_provider=tracer_provider, excluded_urls="/health,/metrics"
            )

        # Instrument libraries
        SQLAlchemyInstrumentor.instrument(tracer_provider=tracer_provider)
        RedisInstrumentor.instrument(tracer_provider=tracer_provider)
        HTTPXClientInstrumentor.instrument(tracer_provider=tracer_provider)

        print(f"✅ OpenTelemetry tracing enabled (sample rate: {self.sample_rate})")

    def get_tracer(self, name: str) -> trace.Tracer:
        """Get tracer for a component."""
        return trace.get_tracer(name)


# Global tracing config
tracing_config = TracingConfig()


def get_tracer(name: str) -> trace.Tracer:
    """Get tracer for a component."""
    return tracing_config.get_tracer(name)


def trace_function(name: Optional[str] = None):
    """Decorator to trace functions."""

    def decorator(func):
        tracer = get_tracer(name or func.__module__)

        async def wrapper(*args, **kwargs):
            with tracer.start_as_span(name or func.__name__) as span:
                # Add function arguments as attributes
                span.set_attribute("function.name", func.__name__)
                span.set_attribute("function.module", func.__module__)

                try:
                    result = await func(*args, **kwargs)
                    span.set_attribute("function.success", True)
                    return result
                except Exception as e:
                    span.set_attribute("function.success", False)
                    span.set_attribute("function.error", str(e))
                    span.record_exception(e)
                    raise

        return wrapper

    return decorator


def add_span_attributes(attributes: Dict[str, any]):
    """Add attributes to current span."""
    span = trace.get_current_span()
    if span:
        for key, value in attributes.items():
            span.set_attribute(key, str(value))


def add_span_event(name: str, attributes: Optional[Dict[str, any]] = None):
    """Add event to current span."""
    span = trace.get_current_span()
    if span:
        span.add_event(name, attributes or {})
