"""Export service for data and results."""

import asyncio
import csv
import json
import io
import zipfile
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import pandas as pd
from fastapi.responses import StreamingResponse

from advance_rag.core.config import get_settings
from advance_rag.core.logging import get_logger

logger = get_logger(__name__)
settings = get_settings()


class ExportFormat(str, Enum):
    """Export formats."""

    JSON = "json"
    CSV = "csv"
    EXCEL = "excel"
    PDF = "pdf"
    MARKDOWN = "markdown"


class ExportService:
    """Service for exporting data and results."""

    def __init__(self):
        """Initialize export service."""
        self.export_dir = Path(settings.EXPORT_DIR or "exports")
        self.export_dir.mkdir(parents=True, exist_ok=True)

    async def export_query_results(
        self,
        query_id: str,
        results: Dict[str, Any],
        format: ExportFormat = ExportFormat.JSON,
        include_sources: bool = True,
    ) -> Union[str, StreamingResponse]:
        """Export query results."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"query_{query_id}_{timestamp}"

        if format == ExportFormat.JSON:
            return await self._export_json(results, filename, include_sources)
        elif format == ExportFormat.CSV:
            return await self._export_csv(results, filename, include_sources)
        elif format == ExportFormat.EXCEL:
            return await self._export_excel(results, filename, include_sources)
        elif format == ExportFormat.MARKDOWN:
            return await self._export_markdown(results, filename, include_sources)
        else:
            raise ValueError(f"Unsupported export format: {format}")

    async def _export_json(
        self, data: Dict[str, Any], filename: str, include_sources: bool
    ) -> str:
        """Export data as JSON."""
        export_data = {
            "query_id": data.get("query_id"),
            "answer": data.get("answer"),
            "mode": data.get("mode"),
            "llm_model": data.get("llm_model"),
            "created_at": data.get("created_at"),
            "prompt_tokens": data.get("prompt_tokens"),
            "completion_tokens": data.get("completion_tokens"),
            "duration_ms": data.get("duration_ms"),
        }

        if include_sources and "sources" in data:
            export_data["sources"] = data["sources"]

        file_path = self.export_dir / f"{filename}.json"

        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False, default=str)

        logger.info(f"Exported JSON: {file_path}")
        return str(file_path)

    async def _export_csv(
        self, data: Dict[str, Any], filename: str, include_sources: bool
    ) -> str:
        """Export data as CSV."""
        file_path = self.export_dir / f"{filename}.csv"

        with open(file_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)

            # Write metadata
            writer.writerow(["Query ID", data.get("query_id", "")])
            writer.writerow(["Mode", data.get("mode", "")])
            writer.writerow(["LLM Model", data.get("llm_model", "")])
            writer.writerow(["Created At", data.get("created_at", "")])
            writer.writerow(["Prompt Tokens", data.get("prompt_tokens", "")])
            writer.writerow(["Completion Tokens", data.get("completion_tokens", "")])
            writer.writerow(["Duration (ms)", data.get("duration_ms", "")])
            writer.writerow([])  # Empty row

            # Write answer
            writer.writerow(["Answer"])
            writer.writerow([data.get("answer", "")])
            writer.writerow([])  # Empty row

            # Write sources
            if include_sources and "sources" in data:
                writer.writerow(["Sources"])
                writer.writerow(
                    ["Chunk ID", "Content", "Score", "Source", "Document ID"]
                )

                for source in data["sources"]:
                    chunk_id = source.get("chunk", {}).get("id", "")
                    content = source.get("chunk", {}).get("content", "")[:200] + "..."
                    score = source.get("score", "")
                    source_type = source.get("source", "")
                    doc_id = source.get("chunk", {}).get("document_id", "")

                    writer.writerow([chunk_id, content, score, source_type, doc_id])

        logger.info(f"Exported CSV: {file_path}")
        return str(file_path)

    async def _export_excel(
        self, data: Dict[str, Any], filename: str, include_sources: bool
    ) -> str:
        """Export data as Excel."""
        file_path = self.export_dir / f"{filename}.xlsx"

        with pd.ExcelWriter(file_path, engine="openpyxl") as writer:
            # Metadata sheet
            metadata_df = pd.DataFrame(
                [
                    ["Query ID", data.get("query_id", "")],
                    ["Mode", data.get("mode", "")],
                    ["LLM Model", data.get("llm_model", "")],
                    ["Created At", data.get("created_at", "")],
                    ["Prompt Tokens", data.get("prompt_tokens", "")],
                    ["Completion Tokens", data.get("completion_tokens", "")],
                    ["Duration (ms)", data.get("duration_ms", "")],
                ],
                columns=["Property", "Value"],
            )
            metadata_df.to_excel(writer, sheet_name="Metadata", index=False)

            # Answer sheet
            answer_df = pd.DataFrame([[data.get("answer", "")]], columns=["Answer"])
            answer_df.to_excel(writer, sheet_name="Answer", index=False)

            # Sources sheet
            if include_sources and "sources" in data:
                sources_data = []
                for source in data["sources"]:
                    sources_data.append(
                        {
                            "Chunk ID": source.get("chunk", {}).get("id", ""),
                            "Content": source.get("chunk", {}).get("content", ""),
                            "Score": source.get("score", ""),
                            "Source": source.get("source", ""),
                            "Document ID": source.get("chunk", {}).get(
                                "document_id", ""
                            ),
                            "Document Type": source.get("chunk", {}).get(
                                "document_type", ""
                            ),
                        }
                    )

                sources_df = pd.DataFrame(sources_data)
                sources_df.to_excel(writer, sheet_name="Sources", index=False)

        logger.info(f"Exported Excel: {file_path}")
        return str(file_path)

    async def _export_markdown(
        self, data: Dict[str, Any], filename: str, include_sources: bool
    ) -> str:
        """Export data as Markdown."""
        file_path = self.export_dir / f"{filename}.md"

        with open(file_path, "w", encoding="utf-8") as f:
            f.write(f"# Query Results\n\n")
            f.write(f"**Query ID:** {data.get('query_id', '')}\n")
            f.write(f"**Mode:** {data.get('mode', '')}\n")
            f.write(f"**LLM Model:** {data.get('llm_model', '')}\n")
            f.write(f"**Created At:** {data.get('created_at', '')}\n")
            f.write(f"**Prompt Tokens:** {data.get('prompt_tokens', '')}\n")
            f.write(f"**Completion Tokens:** {data.get('completion_tokens', '')}\n")
            f.write(f"**Duration:** {data.get('duration_ms', '')} ms\n\n")

            f.write("## Answer\n\n")
            f.write(f"{data.get('answer', '')}\n\n")

            if include_sources and "sources" in data:
                f.write("## Sources\n\n")
                for i, source in enumerate(data["sources"], 1):
                    f.write(f"### Source {i}\n\n")
                    f.write(f"**Score:** {source.get('score', '')}\n")
                    f.write(f"**Source Type:** {source.get('source', '')}\n")
                    f.write(
                        f"**Document ID:** {source.get('chunk', {}).get('document_id', '')}\n"
                    )
                    f.write(
                        f"**Document Type:** {source.get('chunk', {}).get('document_type', '')}\n\n"
                    )
                    f.write(f"**Content:**\n")
                    f.write(f"{source.get('chunk', {}).get('content', '')}\n\n")

        logger.info(f"Exported Markdown: {file_path}")
        return str(file_path)

    async def export_study_data(
        self,
        study_id: str,
        format: ExportFormat = ExportFormat.JSON,
        include_chunks: bool = False,
    ) -> str:
        """Export all data for a study."""
        # This would fetch from database
        # For now, create a placeholder
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"study_{study_id}_{timestamp}"

        study_data = {
            "study_id": study_id,
            "export_date": datetime.utcnow().isoformat(),
            "documents": [],  # Would fetch from DB
            "chunks": [],  # Would fetch from DB if include_chunks
            "entities": [],  # Would fetch from graph
            "relations": [],  # Would fetch from graph
        }

        if format == ExportFormat.JSON:
            return await self._export_json(study_data, filename, True)
        elif format == ExportFormat.EXCEL:
            return await self._export_excel(study_data, filename, True)
        else:
            raise ValueError(f"Unsupported format for study export: {format}")

    async def create_export_package(
        self, study_id: str, formats: List[ExportFormat]
    ) -> StreamingResponse:
        """Create a zip package with multiple export formats."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        package_filename = f"study_{study_id}_export_{timestamp}.zip"

        # Create zip file in memory
        zip_buffer = io.BytesIO()

        with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
            for format in formats:
                try:
                    export_path = await self.export_study_data(
                        study_id=study_id, format=format
                    )

                    # Add file to zip
                    zip_file.write(export_path, Path(export_path).name)

                    # Clean up temporary file
                    Path(export_path).unlink()

                except Exception as e:
                    logger.error(f"Failed to export {format}: {e}")

        zip_buffer.seek(0)

        return StreamingResponse(
            io.BytesIO(zip_buffer.read()),
            media_type="application/zip",
            headers={"Content-Disposition": f"attachment; filename={package_filename}"},
        )

    async def export_audit_log(
        self,
        start_date: datetime,
        end_date: datetime,
        format: ExportFormat = ExportFormat.CSV,
    ) -> str:
        """Export audit log for date range."""
        # This would fetch from database
        audit_data = {
            "start_date": start_date.isoformat(),
            "end_date": end_date.isoformat(),
            "queries": [],  # Would fetch from DB
            "ingestions": [],  # Would fetch from DB
            "errors": [],  # Would fetch from DB
        }

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"audit_log_{timestamp}"

        if format == ExportFormat.JSON:
            return await self._export_json(audit_data, filename, True)
        elif format == ExportFormat.CSV:
            return await self._export_csv(audit_data, filename, True)
        elif format == ExportFormat.EXCEL:
            return await self._export_excel(audit_data, filename, True)
        else:
            raise ValueError(f"Unsupported format for audit log: {format}")


# Global export service
export_service = ExportService()
