#!/usr/bin/env python
"""Test script to verify Rye setup."""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

try:
    from advance_rag.core.config import get_settings
    from advance_rag.core.logging import configure_logging, get_logger

    print("✅ Successfully imported Advance RAG modules")

    # Test configuration
    settings = get_settings()
    print(f"✅ Configuration loaded: {settings.ENVIRONMENT}")

    # Test logging
    configure_logging()
    logger = get_logger("test")
    logger.info("Test logging successful")
    print("✅ Logging configured successfully")

    # Test dummy data generation
    from scripts.generate_dummy_data import ClinicalDataGenerator

    generator = ClinicalDataGenerator(study_id="TEST001", n_subjects=5)
    print("✅ ClinicalDataGenerator imported successfully")

    print("\n🎉 All tests passed! Rye setup is working correctly.")
    print("\nNext steps:")
    print("1. Start databases: docker-compose up -d")
    print("2. Initialize databases: rye run python scripts/init_postgres.py")
    print("3. Start API: rye run uvicorn advance_rag.api.main:app --reload")

except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)
except Exception as e:
    print(f"❌ Error: {e}")
    sys.exit(1)
