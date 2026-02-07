"""Ingestion endpoints."""

import uuid
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, BackgroundTasks, UploadFile, File, Form
from pydantic import BaseModel, Field

from advance_rag.api.dependencies import IngestionServiceDep
from advance_rag.core.logging import get_logger, log_ingestion
from advance_rag.models import IngestionTask

logger = get_logger(__name__)
router = APIRouter()


class IngestionRequest(BaseModel):
    """Request model for ingestion."""

    file_paths: List[str] = Field(..., description="List of file paths to ingest")
    study_id: Optional[str] = Field(None, description="Study ID for organization")
    metadata: Optional[Dict[str, Any]] = Field(
        default_factory=dict, description="Additional metadata"
    )


class IngestionResponse(BaseModel):
    """Response model for ingestion."""

    task_id: str = Field(..., description="Task ID for tracking")
    status: str = Field(..., description="Initial task status")
    message: str = Field(..., description="Status message")


class TaskStatusResponse(BaseModel):
    """Response model for task status."""

    task_id: str = Field(..., description="Task ID")
    status: str = Field(..., description="Task status")
    progress: float = Field(..., description="Progress percentage")
    error_message: Optional[str] = Field(None, description="Error message if failed")
    created_at: str = Field(..., description="Creation timestamp")
    updated_at: str = Field(..., description="Last update timestamp")


# In-memory task storage (in production, use Redis or database)
task_storage: Dict[str, IngestionTask] = {}


@router.post("/files", response_model=IngestionResponse)
async def ingest_files(
    request: IngestionRequest,
    background_tasks: BackgroundTasks,
    ingestion_service: IngestionServiceDep,
) -> IngestionResponse:
    """Ingest files from paths."""
    task_id = str(uuid.uuid4())

    # Create task
    task = IngestionTask(
        id=task_id, file_paths=request.file_paths, status="pending", progress=0.0
    )
    task_storage[task_id] = task

    # Start background ingestion
    background_tasks.add_task(
        _process_ingestion_task, task_id, request.file_paths, ingestion_service
    )

    logger.info(f"Started ingestion task {task_id} for {len(request.file_paths)} files")

    return IngestionResponse(
        task_id=task_id,
        status="pending",
        message=f"Ingestion started for {len(request.file_paths)} files",
    )


@router.post("/upload", response_model=IngestionResponse)
async def upload_and_ingest(
    background_tasks: BackgroundTasks,
    ingestion_service: IngestionServiceDep,
    files: List[UploadFile] = File(...),
    study_id: Optional[str] = Form(None),
) -> IngestionResponse:
    """Upload and ingest files."""
    # Save uploaded files temporarily
    import tempfile
    import os
    from pathlib import Path

    temp_dir = tempfile.mkdtemp()
    file_paths = []

    try:
        for file in files:
            # Validate file
            if not file.filename:
                raise HTTPException(status_code=400, detail="File must have a name")

            # Check file extension
            file_ext = Path(file.filename).suffix.lower()
            if file_ext not in [".json", ".md", ".txt", ".csv"]:
                raise HTTPException(
                    status_code=400, detail=f"Unsupported file type: {file_ext}"
                )

            # Save file
            file_path = os.path.join(temp_dir, file.filename)
            with open(file_path, "wb") as buffer:
                content = await file.read()
                buffer.write(content)

            file_paths.append(file_path)

        # Create ingestion task
        task_id = str(uuid.uuid4())
        task = IngestionTask(
            id=task_id, file_paths=file_paths, status="pending", progress=0.0
        )
        task_storage[task_id] = task

        # Start background ingestion
        background_tasks.add_task(
            _process_ingestion_task, task_id, file_paths, ingestion_service
        )

        logger.info(f"Started upload ingestion task {task_id} for {len(files)} files")

        return IngestionResponse(
            task_id=task_id,
            status="pending",
            message=f"Upload and ingestion started for {len(files)} files",
        )

    except Exception as e:
        # Clean up temp directory
        import shutil

        shutil.rmtree(temp_dir, ignore_errors=True)
        raise


@router.post("/directory", response_model=IngestionResponse)
async def ingest_directory(
    directory_path: str,
    background_tasks: BackgroundTasks,
    ingestion_service: IngestionServiceDep,
    recursive: bool = True,
) -> IngestionResponse:
    """Ingest all files in a directory."""
    from pathlib import Path

    # Validate directory
    path = Path(directory_path)
    if not path.exists():
        raise HTTPException(
            status_code=404, detail=f"Directory not found: {directory_path}"
        )

    if not path.is_dir():
        raise HTTPException(
            status_code=400, detail=f"Path is not a directory: {directory_path}"
        )

    # Create ingestion task
    task_id = str(uuid.uuid4())
    task = IngestionTask(
        id=task_id, file_paths=[directory_path], status="pending", progress=0.0
    )
    task_storage[task_id] = task

    # Start background ingestion
    background_tasks.add_task(
        _process_directory_ingestion_task,
        task_id,
        directory_path,
        recursive,
        ingestion_service,
    )

    logger.info(f"Started directory ingestion task {task_id} for {directory_path}")

    return IngestionResponse(
        task_id=task_id,
        status="pending",
        message=f"Directory ingestion started for {directory_path}",
    )


@router.get("/tasks/{task_id}", response_model=TaskStatusResponse)
async def get_task_status(task_id: str) -> TaskStatusResponse:
    """Get status of an ingestion task."""
    if task_id not in task_storage:
        raise HTTPException(status_code=404, detail="Task not found")

    task = task_storage[task_id]

    return TaskStatusResponse(
        task_id=task.id,
        status=task.status,
        progress=task.progress,
        error_message=task.error_message,
        created_at=task.created_at.isoformat(),
        updated_at=task.updated_at.isoformat(),
    )


@router.get("/tasks")
async def list_tasks(limit: int = 50) -> List[TaskStatusResponse]:
    """List recent ingestion tasks."""
    tasks = list(task_storage.values())
    tasks.sort(key=lambda t: t.created_at, reverse=True)

    return [
        TaskStatusResponse(
            task_id=task.id,
            status=task.status,
            progress=task.progress,
            error_message=task.error_message,
            created_at=task.created_at.isoformat(),
            updated_at=task.updated_at.isoformat(),
        )
        for task in tasks[:limit]
    ]


@router.delete("/tasks/{task_id}")
async def delete_task(task_id: str) -> Dict[str, str]:
    """Delete an ingestion task."""
    if task_id not in task_storage:
        raise HTTPException(status_code=404, detail="Task not found")

    del task_storage[task_id]
    logger.info(f"Deleted task {task_id}")

    return {"message": f"Task {task_id} deleted"}


async def _process_ingestion_task(
    task_id: str, file_paths: List[str], ingestion_service: IngestionServiceDep
) -> None:
    """Process ingestion task in background."""
    import time
    from datetime import datetime

    task = task_storage[task_id]

    try:
        # Update status
        task.status = "processing"
        task.progress = 0.1
        task.updated_at = datetime.utcnow()

        # Ingest files
        document_ids = await ingestion_service.ingest_files(file_paths)

        # Update completion
        task.status = "completed"
        task.progress = 1.0
        task.updated_at = datetime.utcnow()

        logger.info(
            f"Completed ingestion task {task_id}: {len(document_ids)} documents"
        )

        # Log ingestion
        log_ingestion(
            file_path=",".join(file_paths),
            file_size=0,  # Would need to calculate
            num_chunks=0,  # Would need to track
            duration_ms=0,  # Would need to track
            task_id=task_id,
        )

    except Exception as e:
        logger.error(f"Ingestion task {task_id} failed: {e}")
        task.status = "failed"
        task.error_message = str(e)
        task.updated_at = datetime.utcnow()


async def _process_directory_ingestion_task(
    task_id: str,
    directory_path: str,
    recursive: bool,
    ingestion_service: IngestionServiceDep,
) -> None:
    """Process directory ingestion task in background."""
    import time
    from datetime import datetime

    task = task_storage[task_id]

    try:
        # Update status
        task.status = "processing"
        task.progress = 0.1
        task.updated_at = datetime.utcnow()

        # Ingest directory
        document_ids = await ingestion_service.ingest_directory(
            directory_path, recursive
        )

        # Update completion
        task.status = "completed"
        task.progress = 1.0
        task.updated_at = datetime.utcnow()

        logger.info(
            f"Completed directory ingestion task {task_id}: {len(document_ids)} documents"
        )

    except Exception as e:
        logger.error(f"Directory ingestion task {task_id} failed: {e}")
        task.status = "failed"
        task.error_message = str(e)
        task.updated_at = datetime.utcnow()
