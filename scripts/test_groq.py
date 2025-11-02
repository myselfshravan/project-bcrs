#!/usr/bin/env python3
"""Quick test of Groq LLM integration."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from tcris.llm.groq_service import GroqService
from loguru import logger

def test_groq_service():
    """Test Groq service initialization and basic functionality."""
    logger.info("Testing Groq LLM service...")

    try:
        # Initialize service
        logger.info("1. Initializing Groq service...")
        groq = GroqService()
        logger.info("   ✓ Service initialized successfully")

        # Test prediction explanation
        logger.info("2. Testing prediction explanation...")
        explanation = groq.explain_prediction(
            baseline_tumors=3,
            baseline_size=2.5,
            treatment="placebo",
            time_horizon=24,
            cox_risk=0.85,
            recurrence_prob=45.2,
            risk_level="Moderate"
        )
        logger.info("   ✓ Generated explanation:")
        logger.info(f"   {explanation[:200]}...")

        # Test treatment explanation
        logger.info("3. Testing treatment explanation...")
        simple = groq.explain_treatment_simple(
            recommended="thiotepa",
            risk_reduction=25.5,
            survival_improvement=15.3
        )
        logger.info("   ✓ Generated treatment explanation:")
        logger.info(f"   {simple}")

        logger.info("\n✅ All Groq integration tests PASSED!")
        return True

    except Exception as e:
        logger.error(f"❌ Groq test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_groq_service()
    sys.exit(0 if success else 1)
