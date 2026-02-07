"""Notification service for alerts and updates."""

import asyncio
import json
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from advance_rag.core.config import get_settings
from advance_rag.core.logging import get_logger

logger = get_logger(__name__)
settings = get_settings()


class NotificationType(str, Enum):
    """Notification types."""

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    SUCCESS = "success"
    SYSTEM = "system"


class NotificationChannel(str, Enum):
    """Notification channels."""

    EMAIL = "email"
    WEBHOOK = "webhook"
    IN_APP = "in_app"
    SLACK = "slack"
    TEAMS = "teams"


class Notification:
    """Notification model."""

    def __init__(
        self,
        id: str,
        type: NotificationType,
        title: str,
        message: str,
        channels: List[NotificationChannel],
        recipients: List[str],
        metadata: Optional[Dict[str, Any]] = None,
        created_at: Optional[datetime] = None,
    ):
        self.id = id
        self.type = type
        self.title = title
        self.message = message
        self.channels = channels
        self.recipients = recipients
        self.metadata = metadata or {}
        self.created_at = created_at or datetime.utcnow()
        self.sent = False
        self.sent_at = None
        self.error = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "type": self.type.value,
            "title": self.title,
            "message": self.message,
            "channels": [c.value for c in self.channels],
            "recipients": self.recipients,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat(),
            "sent": self.sent,
            "sent_at": self.sent_at.isoformat() if self.sent_at else None,
            "error": self.error,
        }


class EmailProvider:
    """Email notification provider."""

    def __init__(self):
        self.smtp_server = settings.SMTP_SERVER
        self.smtp_port = settings.SMTP_PORT
        self.smtp_username = settings.SMTP_USERNAME
        self.smtp_password = settings.SMTP_PASSWORD
        self.from_email = settings.FROM_EMAIL

    async def send(self, notification: Notification) -> bool:
        """Send email notification."""
        try:
            import smtplib
            from email.mime.text import MIMEText
            from email.mime.multipart import MIMEMultipart

            # Create message
            msg = MIMEMultipart()
            msg["From"] = self.from_email
            msg["To"] = ", ".join(notification.recipients)
            msg["Subject"] = notification.title

            # Add body
            body = MIMEText(notification.message, "html")
            msg.attach(body)

            # Send email
            server = smtplib.SMTP(self.smtp_server, self.smtp_port)
            server.starttls()
            server.login(self.smtp_username, self.smtp_password)
            server.send_message(msg)
            server.quit()

            return True
        except Exception as e:
            logger.error(f"Failed to send email: {e}")
            return False


class WebhookProvider:
    """Webhook notification provider."""

    def __init__(self):
        self.webhook_urls = settings.WEBHOOK_URLS or {}

    async def send(self, notification: Notification) -> bool:
        """Send webhook notification."""
        try:
            import httpx

            payload = {
                "id": notification.id,
                "type": notification.type.value,
                "title": notification.title,
                "message": notification.message,
                "timestamp": notification.created_at.isoformat(),
                "metadata": notification.metadata,
            }

            async with httpx.AsyncClient() as client:
                for url in self.webhook_urls.values():
                    response = await client.post(url, json=payload, timeout=30.0)
                    response.raise_for_status()

            return True
        except Exception as e:
            logger.error(f"Failed to send webhook: {e}")
            return False


class InAppProvider:
    """In-app notification provider."""

    def __init__(self):
        self.notifications: Dict[str, List[Notification]] = {}

    async def send(self, notification: Notification) -> bool:
        """Store in-app notification."""
        try:
            for recipient in notification.recipients:
                if recipient not in self.notifications:
                    self.notifications[recipient] = []
                self.notifications[recipient].append(notification)

            return True
        except Exception as e:
            logger.error(f"Failed to store in-app notification: {e}")
            return False

    def get_user_notifications(
        self, user_id: str, limit: int = 50
    ) -> List[Notification]:
        """Get notifications for user."""
        user_notifications = self.notifications.get(user_id, [])
        # Sort by created_at descending
        user_notifications.sort(key=lambda n: n.created_at, reverse=True)
        return user_notifications[:limit]


class NotificationService:
    """Main notification service."""

    def __init__(self):
        """Initialize notification service."""
        self.providers = {
            NotificationChannel.EMAIL: EmailProvider(),
            NotificationChannel.WEBHOOK: WebhookProvider(),
            NotificationChannel.IN_APP: InAppProvider(),
        }
        self.notification_queue: asyncio.Queue = asyncio.Queue()
        self.task = None
        self.running = False

    async def start(self):
        """Start notification processor."""
        if not self.running:
            self.running = True
            self.task = asyncio.create_task(self._process_notifications())
            logger.info("Notification service started")

    async def stop(self):
        """Stop notification service."""
        if self.running:
            self.running = False
            if self.task:
                self.task.cancel()
                try:
                    await self.task
                except asyncio.CancelledError:
                    pass
            logger.info("Notification service stopped")

    async def send_notification(
        self,
        type: NotificationType,
        title: str,
        message: str,
        channels: List[NotificationChannel],
        recipients: List[str],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Send notification."""
        import uuid

        notification = Notification(
            id=str(uuid.uuid4()),
            type=type,
            title=title,
            message=message,
            channels=channels,
            recipients=recipients,
            metadata=metadata,
        )

        await self.notification_queue.put(notification)
        return notification.id

    async def _process_notifications(self):
        """Process notifications from queue."""
        while self.running:
            try:
                notification = await asyncio.wait_for(
                    self.notification_queue.get(), timeout=1.0
                )

                await self._send_notification(notification)

            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Error processing notification: {e}")

    async def _send_notification(self, notification: Notification):
        """Send notification through all channels."""
        for channel in notification.channels:
            if channel in self.providers:
                provider = self.providers[channel]
                success = await provider.send(notification)

                if success:
                    logger.info(f"Sent {channel.value} notification: {notification.id}")
                else:
                    logger.error(
                        f"Failed to send {channel.value} notification: {notification.id}"
                    )
                    notification.error = f"Failed to send via {channel.value}"

        notification.sent = True
        notification.sent_at = datetime.utcnow()

    def get_user_notifications(
        self, user_id: str, limit: int = 50
    ) -> List[Dict[str, Any]]:
        """Get user's in-app notifications."""
        in_app_provider = self.providers.get(NotificationChannel.IN_APP)
        if isinstance(in_app_provider, InAppProvider):
            notifications = in_app_provider.get_user_notifications(user_id, limit)
            return [n.to_dict() for n in notifications]
        return []

    async def send_ingestion_complete(
        self, task_id: str, file_count: int, user_id: str
    ):
        """Send ingestion completion notification."""
        await self.send_notification(
            type=NotificationType.SUCCESS,
            title="Data Ingestion Complete",
            message=f"Successfully ingested {file_count} files. Task ID: {task_id}",
            channels=[NotificationChannel.IN_APP, NotificationChannel.EMAIL],
            recipients=[user_id],
            metadata={"task_id": task_id, "file_count": file_count},
        )

    async def send_ingestion_error(self, task_id: str, error: str, user_id: str):
        """Send ingestion error notification."""
        await self.send_notification(
            type=NotificationType.ERROR,
            title="Data Ingestion Failed",
            message=f"Failed to ingest data. Task ID: {task_id}. Error: {error}",
            channels=[NotificationChannel.IN_APP, NotificationChannel.EMAIL],
            recipients=[user_id],
            metadata={"task_id": task_id, "error": error},
        )

    async def send_system_alert(
        self, title: str, message: str, metadata: Optional[Dict[str, Any]] = None
    ):
        """Send system alert."""
        await self.send_notification(
            type=NotificationType.SYSTEM,
            title=title,
            message=message,
            channels=[NotificationChannel.WEBHOOK, NotificationChannel.EMAIL],
            recipients=["admin@example.com"],  # Would get from config
            metadata=metadata,
        )


# Global notification service
notification_service = NotificationService()
