"""Celery configuration for background tasks."""

from celery import Celery
from advance_rag.core.config import get_settings

settings = get_settings()

# Create Celery app
celery_app = Celery(
    "advance_rag",
    broker=settings.CELERY_BROKER_URL,
    backend=settings.CELERY_RESULT_BACKEND,
    include=["advance_rag.tasks"],
)

# Configure Celery
celery_app.conf.update(
    task_serializer=settings.CELERY_TASK_SERIALIZER,
    result_serializer=settings.CELERY_RESULT_SERIALIZER,
    accept_content=settings.CELERY_ACCEPT_CONTENT,
    timezone=settings.CELERY_TIMEZONE,
    enable_utc=True,
    task_track_started=True,
    task_time_limit=30 * 60,  # 30 minutes
    task_soft_time_limit=25 * 60,  # 25 minutes
    worker_prefetch_multiplier=1,
    worker_max_tasks_per_child=1000,
)

# Configure task routes
celery_app.conf.task_routes = {
    "advance_rag.tasks.ingestion.*": {"queue": "ingestion"},
    "advance_rag.tasks.embedding.*": {"queue": "embedding"},
    "advance_rag.tasks.graph.*": {"queue": "graph"},
}

# Configure beat schedule for periodic tasks
celery_app.conf.beat_schedule = {
    "cleanup-old-tasks": {
        "task": "advance_rag.tasks.cleanup.cleanup_old_tasks",
        "schedule": 3600.0,  # Every hour
    },
    "update-communities": {
        "task": "advance_rag.tasks.graph.update_communities",
        "schedule": 86400.0,  # Daily
    },
}
