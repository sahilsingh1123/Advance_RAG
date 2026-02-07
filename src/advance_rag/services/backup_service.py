"""Backup service for data and configuration."""

import asyncio
import json
import shutil
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

import asyncpg
import aiofiles
from neo4j import GraphDatabase

from advance_rag.core.config import get_settings
from advance_rag.core.logging import get_logger

logger = get_logger(__name__)
settings = get_settings()


class BackupService:
    """Service for creating and managing backups."""

    def __init__(self):
        """Initialize backup service."""
        self.backup_dir = Path(settings.BACKUP_DIR or "backups")
        self.backup_dir.mkdir(parents=True, exist_ok=True)
        self.retention_days = settings.BACKUP_RETENTION_DAYS or 30

    async def create_full_backup(self, study_id: Optional[str] = None) -> str:
        """Create a full backup of the system."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_name = f"full_backup_{timestamp}"
        if study_id:
            backup_name = f"study_{study_id}_backup_{timestamp}"

        backup_path = self.backup_dir / backup_name
        backup_path.mkdir(parents=True, exist_ok=True)

        logger.info(f"Starting full backup: {backup_name}")

        try:
            # Backup PostgreSQL
            await self._backup_postgresql(backup_path)

            # Backup Neo4j
            await self._backup_neo4j(backup_path)

            # Backup configuration
            await self._backup_config(backup_path)

            # Backup files
            await self._backup_files(backup_path)

            # Create backup metadata
            await self._create_backup_metadata(backup_path, study_id)

            logger.info(f"Full backup completed: {backup_path}")
            return str(backup_path)

        except Exception as e:
            logger.error(f"Backup failed: {e}")
            # Cleanup failed backup
            if backup_path.exists():
                shutil.rmtree(backup_path)
            raise

    async def _backup_postgresql(self, backup_path: Path):
        """Backup PostgreSQL database."""
        pg_backup_path = backup_path / "postgresql"
        pg_backup_path.mkdir(exist_ok=True)

        # Use pg_dump
        import subprocess

        cmd = [
            "pg_dump",
            f"--dbname={settings.DATABASE_URL}",
            "--format=custom",
            "--compress=9",
            "--file",
            str(pg_backup_path / "database.dump"),
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            raise Exception(f"PostgreSQL backup failed: {result.stderr}")

        logger.info("PostgreSQL backup completed")

    async def _backup_neo4j(self, backup_path: Path):
        """Backup Neo4j database."""
        neo4j_backup_path = backup_path / "neo4j"
        neo4j_backup_path.mkdir(exist_ok=True)

        driver = GraphDatabase.driver(
            settings.NEO4J_URI, auth=(settings.NEO4J_USER, settings.NEO4J_PASSWORD)
        )

        try:
            with driver.session(database="neo4j") as session:
                # Export all data as Cypher
                result = session.run(
                    "CALL apoc.export.cypher.all('neo4j/data.cypher', {format: 'cypher-shell'})"
                )

                # Move the exported file to backup location
                neo4j_data_dir = Path(
                    "/var/lib/neo4j/import"
                )  # Default Neo4j import directory
                source_file = neo4j_data_dir / "data.cypher"
                if source_file.exists():
                    shutil.move(
                        str(source_file), str(neo4j_backup_path / "data.cypher")
                    )

                # Export schema
                schema_file = neo4j_backup_path / "schema.cypher"
                with open(schema_file, "w") as f:
                    f.write("-- Neo4j Schema Backup\n")
                    f.write(f"-- Created: {datetime.utcnow().isoformat()}\n\n")

                    # Export constraints
                    constraints = session.run("SHOW CONSTRAINTS")
                    f.write("-- Constraints\n")
                    for record in constraints:
                        f.write(f"{record['name']}\n")

                    # Export indexes
                    indexes = session.run("SHOW INDEXES")
                    f.write("\n-- Indexes\n")
                    for record in indexes:
                        f.write(f"{record['name']}\n")

            logger.info("Neo4j backup completed")

        finally:
            driver.close()

    async def _backup_config(self, backup_path: Path):
        """Backup configuration files."""
        config_backup_path = backup_path / "config"
        config_backup_path.mkdir(exist_ok=True)

        # Backup environment variables
        env_backup = config_backup_path / "env.txt"
        async with aiofiles.open(env_backup, "w") as f:
            await f.write("# Environment Variables Backup\n")
            await f.write(f"# Created: {datetime.utcnow().isoformat()}\n\n")

            # List important env vars (without values for security)
            important_vars = [
                "DATABASE_URL",
                "REDIS_URL",
                "NEO4J_URI",
                "OPENAI_API_KEY",
                "ANTHROPIC_API_KEY",
                "SECRET_KEY",
                "ENVIRONMENT",
            ]

            for var in important_vars:
                value = getattr(settings, var, None)
                if value:
                    # Mask sensitive values
                    if "KEY" in var or "PASSWORD" in var:
                        value = "***MASKED***"
                    await f.write(f"{var}={value}\n")

        # Copy configuration files
        config_files = [
            "docker-compose.yml",
            "config/nginx.conf",
            "config/prometheus.yml",
        ]

        for config_file in config_files:
            source = Path(config_file)
            if source.exists():
                dest = config_backup_path / source.name
                shutil.copy2(source, dest)

        logger.info("Configuration backup completed")

    async def _backup_files(self, backup_path: Path):
        """Backup important files."""
        files_backup_path = backup_path / "files"
        files_backup_path.mkdir(exist_ok=True)

        # Backup data directory (excluding raw data)
        data_dir = Path("data")
        if data_dir.exists():
            backup_data_dir = files_backup_path / "data"
            shutil.copytree(
                data_dir, backup_data_dir, ignore=shutil.ignore_patterns("raw/*")
            )

        # Backup logs
        logs_dir = Path("logs")
        if logs_dir.exists():
            backup_logs_dir = files_backup_path / "logs"
            shutil.copytree(logs_dir, backup_logs_dir)

        logger.info("Files backup completed")

    async def _create_backup_metadata(self, backup_path: Path, study_id: Optional[str]):
        """Create backup metadata file."""
        metadata = {
            "backup_type": "full",
            "study_id": study_id,
            "created_at": datetime.utcnow().isoformat(),
            "version": "0.1.0",
            "components": {
                "postgresql": True,
                "neo4j": True,
                "config": True,
                "files": True,
            },
            "size_bytes": self._get_directory_size(backup_path),
        }

        metadata_file = backup_path / "metadata.json"
        async with aiofiles.open(metadata_file, "w") as f:
            await f.write(json.dumps(metadata, indent=2))

    def _get_directory_size(self, path: Path) -> int:
        """Get total size of directory in bytes."""
        total_size = 0
        for file_path in path.rglob("*"):
            if file_path.is_file():
                total_size += file_path.stat().st_size
        return total_size

    async def restore_backup(self, backup_path: str) -> bool:
        """Restore from backup."""
        backup_dir = Path(backup_path)

        if not backup_dir.exists():
            raise FileNotFoundError(f"Backup not found: {backup_path}")

        # Read metadata
        metadata_file = backup_dir / "metadata.json"
        if not metadata_file.exists():
            raise FileNotFoundError("Backup metadata not found")

        async with aiofiles.open(metadata_file, "r") as f:
            metadata = json.loads(await f.read())

        logger.info(f"Restoring backup from {metadata['created_at']}")

        try:
            # Restore PostgreSQL
            if metadata["components"]["postgresql"]:
                await self._restore_postgresql(backup_dir / "postgresql")

            # Restore Neo4j
            if metadata["components"]["neo4j"]:
                await self._restore_neo4j(backup_dir / "neo4j")

            # Restore configuration
            if metadata["components"]["config"]:
                await self._restore_config(backup_dir / "config")

            logger.info("Backup restore completed")
            return True

        except Exception as e:
            logger.error(f"Restore failed: {e}")
            raise

    async def _restore_postgresql(self, backup_path: Path):
        """Restore PostgreSQL database."""
        dump_file = backup_path / "database.dump"

        if not dump_file.exists():
            raise FileNotFoundError("PostgreSQL dump file not found")

        import subprocess

        cmd = [
            "pg_restore",
            f"--dbname={settings.DATABASE_URL}",
            "--clean",
            "--if-exists",
            "--verbose",
            str(dump_file),
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            raise Exception(f"PostgreSQL restore failed: {result.stderr}")

        logger.info("PostgreSQL restore completed")

    async def _restore_neo4j(self, backup_path: Path):
        """Restore Neo4j database."""
        data_file = backup_path / "data.cypher"
        schema_file = backup_path / "schema.cypher"

        driver = GraphDatabase.driver(
            settings.NEO4J_URI, auth=(settings.NEO4J_USER, settings.NEO4J_PASSWORD)
        )

        try:
            with driver.session(database="neo4j") as session:
                # Clear existing data
                session.run("MATCH (n) DETACH DELETE n")

                # Restore schema
                if schema_file.exists():
                    with open(schema_file, "r") as f:
                        schema_commands = f.read().split(";")
                        for command in schema_commands:
                            command = command.strip()
                            if command and not command.startswith("--"):
                                try:
                                    session.run(command)
                                except Exception as e:
                                    logger.warning(f"Schema command failed: {e}")

                # Restore data
                if data_file.exists():
                    # Copy to Neo4j import directory
                    neo4j_import_dir = Path("/var/lib/neo4j/import")
                    dest_file = neo4j_import_dir / "restore.cypher"
                    shutil.copy2(data_file, dest_file)

                    # Run import
                    session.run("CALL apoc.cypher.runFile('restore.cypher', {})")

            logger.info("Neo4j restore completed")

        finally:
            driver.close()

    async def _restore_config(self, backup_path: Path):
        """Restore configuration files."""
        # Configuration restore is manual for security
        logger.warning("Configuration restore must be done manually for security")

    async def cleanup_old_backups(self):
        """Clean up old backups based on retention policy."""
        cutoff_date = datetime.utcnow() - timedelta(days=self.retention_days)

        for backup_dir in self.backup_dir.iterdir():
            if backup_dir.is_dir():
                # Check creation date from metadata
                metadata_file = backup_dir / "metadata.json"
                if metadata_file.exists():
                    async with aiofiles.open(metadata_file, "r") as f:
                        metadata = json.loads(await f.read())

                    created_at = datetime.fromisoformat(metadata["created_at"])

                    if created_at < cutoff_date:
                        logger.info(f"Deleting old backup: {backup_dir.name}")
                        shutil.rmtree(backup_dir)

    async def list_backups(self) -> List[Dict]:
        """List all available backups."""
        backups = []

        for backup_dir in self.backup_dir.iterdir():
            if backup_dir.is_dir():
                metadata_file = backup_dir / "metadata.json"

                if metadata_file.exists():
                    async with aiofiles.open(metadata_file, "r") as f:
                        metadata = json.loads(await f.read())

                    backups.append(
                        {
                            "name": backup_dir.name,
                            "path": str(backup_dir),
                            "type": metadata.get("backup_type", "unknown"),
                            "study_id": metadata.get("study_id"),
                            "created_at": metadata.get("created_at"),
                            "size_bytes": metadata.get("size_bytes", 0),
                        }
                    )

        # Sort by creation date descending
        backups.sort(key=lambda x: x["created_at"], reverse=True)

        return backups

    async def schedule_backups(self):
        """Schedule automatic backups."""
        while True:
            try:
                # Create daily backup at 2 AM
                now = datetime.utcnow()
                next_backup = now.replace(hour=2, minute=0, second=0, microsecond=0)

                if next_backup <= now:
                    next_backup += timedelta(days=1)

                sleep_seconds = (next_backup - now).total_seconds()
                await asyncio.sleep(sleep_seconds)

                # Create backup
                await self.create_full_backup()

                # Clean up old backups
                await self.cleanup_old_backups()

            except Exception as e:
                logger.error(f"Scheduled backup failed: {e}")
                await asyncio.sleep(3600)  # Retry in 1 hour


# Global backup service
backup_service = BackupService()
