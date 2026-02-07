"""Utility functions for the RAG system."""

import re
import hashlib
import uuid
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Union

import numpy as np


def generate_id(prefix: str = "id") -> str:
    """Generate a unique ID with prefix."""
    return f"{prefix}_{uuid.uuid4().hex[:16]}"


def hash_text(text: str) -> str:
    """Generate SHA256 hash of text."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def clean_text(text: str) -> str:
    """Clean and normalize text."""
    # Remove extra whitespace
    text = re.sub(r"\s+", " ", text)
    # Remove control characters
    text = re.sub(r"[\x00-\x1f\x7f-\x9f]", "", text)
    # Normalize whitespace
    text = text.strip()
    return text


def truncate_text(text: str, max_length: int, suffix: str = "...") -> str:
    """Truncate text to maximum length."""
    if len(text) <= max_length:
        return text
    return text[: max_length - len(suffix)] + suffix


def extract_keywords(text: str, min_length: int = 3) -> List[str]:
    """Extract keywords from text."""
    # Simple keyword extraction
    words = re.findall(r"\b[a-zA-Z]{" + str(min_length) + ",}\b", text.lower())
    # Remove common words
    stop_words = {
        "the",
        "and",
        "for",
        "are",
        "but",
        "not",
        "you",
        "all",
        "can",
        "had",
        "her",
        "was",
        "one",
        "our",
        "out",
        "day",
        "get",
        "has",
        "him",
        "his",
        "how",
        "man",
        "new",
        "now",
        "old",
        "see",
        "two",
        "way",
        "who",
        "boy",
        "did",
        "its",
        "let",
        "put",
        "say",
        "she",
        "too",
        "use",
    }
    keywords = [word for word in words if word not in stop_words]
    # Return unique keywords
    return list(set(keywords))


def normalize_score(score: float, min_val: float = 0.0, max_val: float = 1.0) -> float:
    """Normalize score to [0, 1] range."""
    if max_val == min_val:
        return 0.0
    return (score - min_val) / (max_val - min_val)


def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """Safe division with default value."""
    if denominator == 0:
        return default
    return numerator / denominator


def format_bytes(bytes_count: int) -> str:
    """Format bytes in human readable format."""
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if bytes_count < 1024.0:
            return f"{bytes_count:.2f} {unit}"
        bytes_count /= 1024.0
    return f"{bytes_count:.2f} PB"


def format_duration(seconds: float) -> str:
    """Format duration in human readable format."""
    if seconds < 1:
        return f"{seconds * 1000:.0f}ms"
    elif seconds < 60:
        return f"{seconds:.2f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.2f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.2f}h"


def merge_dicts(dict1: Dict[str, Any], dict2: Dict[str, Any]) -> Dict[str, Any]:
    """Merge two dictionaries recursively."""
    result = dict1.copy()
    for key, value in dict2.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = merge_dicts(result[key], value)
        else:
            result[key] = value
    return result


def flatten_dict(
    d: Dict[str, Any], parent_key: str = "", sep: str = "."
) -> Dict[str, Any]:
    """Flatten nested dictionary."""
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def chunk_list(lst: List[Any], chunk_size: int) -> List[List[Any]]:
    """Split list into chunks."""
    return [lst[i : i + chunk_size] for i in range(0, len(lst), chunk_size)]


def retry_on_exception(max_retries: int = 3, delay: float = 1.0, backoff: float = 2.0):
    """Decorator for retrying functions on exception."""

    def decorator(func):
        async def wrapper(*args, **kwargs):
            retries = 0
            current_delay = delay
            last_exception = None

            while retries < max_retries:
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    retries += 1
                    if retries >= max_retries:
                        break
                    await asyncio.sleep(current_delay)
                    current_delay *= backoff

            raise last_exception

        return wrapper

    return decorator


def validate_email(email: str) -> bool:
    """Validate email format."""
    pattern = r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"
    return bool(re.match(pattern, email))


def sanitize_filename(filename: str) -> str:
    """Sanitize filename for safe storage."""
    # Remove invalid characters
    filename = re.sub(r'[<>:"/\\|?*]', "_", filename)
    # Remove control characters
    filename = re.sub(r"[\x00-\x1f\x7f-\x9f]", "", filename)
    # Limit length
    if len(filename) > 255:
        name, ext = filename.rsplit(".", 1) if "." in filename else (filename, "")
        filename = name[: 255 - len(ext) - 1] + "." + ext if ext else name[:255]
    return filename


def parse_date_string(date_str: str) -> Optional[datetime]:
    """Parse date string in various formats."""
    formats = [
        "%Y-%m-%d",
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%dT%H:%M:%SZ",
        "%m/%d/%Y",
        "%d/%m/%Y",
    ]

    for fmt in formats:
        try:
            return datetime.strptime(date_str, fmt)
        except ValueError:
            continue

    return None


def calculate_age(
    birth_date: datetime, reference_date: Optional[datetime] = None
) -> int:
    """Calculate age from birth date."""
    if reference_date is None:
        reference_date = datetime.now()

    age = reference_date.year - birth_date.year
    if reference_date.month < birth_date.month or (
        reference_date.month == birth_date.month and reference_date.day < birth_date.day
    ):
        age -= 1

    return age


def detect_phi(text: str) -> List[Dict[str, Any]]:
    """Detect potential PHI in text (basic implementation)."""
    phi_patterns = [
        (r"\b\d{3}-\d{2}-\d{4}\b", "SSN"),
        (r"\b\d{3}-\d{3}-\d{4}\b", "Phone"),
        (r"\b[A-Z]{2}\d{4}\b", "Medical Record"),
        (r"\b\d{2}/\d{2}/\d{4}\b", "Date"),
        (r"\b\d{4}-\d{2}-\d{2}\b", "Date"),
        (r"\b[A-Za-z]+,\s*[A-Za-z]+\b", "Name"),
    ]

    phi_matches = []
    for pattern, phi_type in phi_patterns:
        matches = re.finditer(pattern, text)
        for match in matches:
            phi_matches.append(
                {
                    "type": phi_type,
                    "text": match.group(),
                    "start": match.start(),
                    "end": match.end(),
                }
            )

    return phi_matches


def redact_phi(text: str, replacement: str = "[REDACTED]") -> str:
    """Redact PHI from text."""
    phi_matches = detect_phi(text)

    # Work backwards to avoid index shifting
    for match in reversed(phi_matches):
        text = text[: match["start"]] + replacement + text[match["end"] :]

    return text


def calculate_similarity_scores(
    query_embedding: List[float], document_embeddings: List[List[float]]
) -> List[float]:
    """Calculate cosine similarity scores."""
    if not document_embeddings:
        return []

    query_vec = np.array(query_embedding)
    doc_vecs = np.array(document_embeddings)

    # Calculate cosine similarity
    dot_products = np.dot(doc_vecs, query_vec)
    query_norm = np.linalg.norm(query_vec)
    doc_norms = np.linalg.norm(doc_vecs, axis=1)

    # Avoid division by zero
    similarities = np.divide(
        dot_products,
        query_norm * doc_norms,
        out=np.zeros_like(dot_products),
        where=(query_norm * doc_norms) != 0,
    )

    return similarities.tolist()


def create_batches(items: List[Any], batch_size: int) -> List[List[Any]]:
    """Create batches from items."""
    return [items[i : i + batch_size] for i in range(0, len(items), batch_size)]


def get_file_extension(filename: str) -> str:
    """Get file extension in lowercase."""
    return filename.split(".")[-1].lower() if "." in filename else ""


def is_json_serializable(obj: Any) -> bool:
    """Check if object is JSON serializable."""
    try:
        import json

        json.dumps(obj)
        return True
    except (TypeError, ValueError):
        return False


def deep_get(dictionary: Dict[str, Any], keys: str, default: Any = None) -> Any:
    """Get value from nested dictionary using dot notation."""
    keys = keys.split(".")
    value = dictionary

    for key in keys:
        if isinstance(value, dict) and key in value:
            value = value[key]
        else:
            return default

    return value


def deep_set(dictionary: Dict[str, Any], keys: str, value: Any) -> None:
    """Set value in nested dictionary using dot notation."""
    keys = keys.split(".")
    current = dictionary

    for key in keys[:-1]:
        if key not in current:
            current[key] = {}
        current = current[key]

    current[keys[-1]] = value
