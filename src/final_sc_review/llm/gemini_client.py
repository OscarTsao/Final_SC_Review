"""Gemini API client for LLM integration experiments.

This module provides a clean interface to Gemini 1.5 Flash for:
- LLM reranking (post-P3)
- LLM verification (evidence correctness)
"""

import json
import logging
import os
import time
from typing import Any, Dict, List, Optional

import google.generativeai as genai

logger = logging.getLogger(__name__)


class GeminiClient:
    """Wrapper for Gemini API with retry logic and error handling."""

    def __init__(
        self,
        model_name: str = "gemini-1.5-flash",
        api_key: Optional[str] = None,
        temperature: float = 0.0,
        max_retries: int = 3,
        retry_delay: float = 1.0,
    ):
        """Initialize Gemini client.

        Args:
            model_name: Gemini model name (default: gemini-1.5-flash)
            api_key: API key (if None, reads from GEMINI_API_KEY env var)
            temperature: Sampling temperature (0.0 for deterministic)
            max_retries: Maximum retry attempts on failure
            retry_delay: Delay between retries in seconds
        """
        self.model_name = model_name
        self.temperature = temperature
        self.max_retries = max_retries
        self.retry_delay = retry_delay

        # Configure API key
        api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError(
                "GEMINI_API_KEY not found. Set it via environment variable or pass as argument."
            )

        genai.configure(api_key=api_key)

        # Initialize model
        self.model = genai.GenerativeModel(
            model_name=model_name,
            generation_config={
                "temperature": temperature,
                "top_p": 0.95,
                "top_k": 40,
                "max_output_tokens": 8192,
            },
        )

        logger.info(f"Initialized GeminiClient with model={model_name}, temp={temperature}")

    def generate_json(
        self,
        prompt: str,
        schema: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Generate JSON response from Gemini.

        Args:
            prompt: Input prompt
            schema: Optional JSON schema for validation

        Returns:
            Parsed JSON response

        Raises:
            ValueError: If response is not valid JSON
            Exception: If API call fails after retries
        """
        for attempt in range(self.max_retries):
            try:
                # Generate response
                response = self.model.generate_content(prompt)

                # Extract text
                text = response.text.strip()

                # Try to extract JSON from code blocks
                if "```json" in text:
                    start = text.find("```json") + 7
                    end = text.find("```", start)
                    text = text[start:end].strip()
                elif "```" in text:
                    start = text.find("```") + 3
                    end = text.find("```", start)
                    text = text[start:end].strip()

                # Parse JSON
                result = json.loads(text)

                # Validate schema if provided
                if schema:
                    self._validate_schema(result, schema)

                return result

            except json.JSONDecodeError as e:
                logger.warning(
                    f"Attempt {attempt + 1}/{self.max_retries}: JSON decode error: {e}"
                )
                if attempt == self.max_retries - 1:
                    logger.error(f"Raw response: {text}")
                    raise ValueError(f"Failed to parse JSON after {self.max_retries} attempts")

            except Exception as e:
                logger.warning(f"Attempt {attempt + 1}/{self.max_retries}: API error: {e}")
                if attempt == self.max_retries - 1:
                    raise

            # Exponential backoff
            if attempt < self.max_retries - 1:
                delay = self.retry_delay * (2**attempt)
                logger.info(f"Retrying in {delay}s...")
                time.sleep(delay)

    def generate_text(self, prompt: str) -> str:
        """Generate text response from Gemini.

        Args:
            prompt: Input prompt

        Returns:
            Generated text

        Raises:
            Exception: If API call fails after retries
        """
        for attempt in range(self.max_retries):
            try:
                response = self.model.generate_content(prompt)
                return response.text.strip()

            except Exception as e:
                logger.warning(f"Attempt {attempt + 1}/{self.max_retries}: API error: {e}")
                if attempt == self.max_retries - 1:
                    raise

            # Exponential backoff
            if attempt < self.max_retries - 1:
                delay = self.retry_delay * (2**attempt)
                logger.info(f"Retrying in {delay}s...")
                time.sleep(delay)

    @staticmethod
    def _validate_schema(data: Dict[str, Any], schema: Dict[str, Any]) -> None:
        """Basic schema validation (checks required keys and types)."""
        if "required" in schema:
            for key in schema["required"]:
                if key not in data:
                    raise ValueError(f"Missing required key: {key}")

        if "properties" in schema:
            for key, prop_schema in schema["properties"].items():
                if key in data:
                    expected_type = prop_schema.get("type")
                    actual_value = data[key]

                    # Type checking (basic)
                    type_map = {
                        "string": str,
                        "number": (int, float),
                        "integer": int,
                        "boolean": bool,
                        "array": list,
                        "object": dict,
                    }

                    if expected_type in type_map:
                        expected_python_type = type_map[expected_type]
                        if not isinstance(actual_value, expected_python_type):
                            raise ValueError(
                                f"Invalid type for {key}: expected {expected_type}, "
                                f"got {type(actual_value).__name__}"
                            )


def test_gemini_connection():
    """Test Gemini API connection."""
    try:
        client = GeminiClient()
        result = client.generate_json(
            prompt="""
            Return a JSON object with the following structure:
            {
                "status": "ok",
                "message": "Connection successful"
            }
            """,
            schema={
                "type": "object",
                "required": ["status", "message"],
                "properties": {
                    "status": {"type": "string"},
                    "message": {"type": "string"},
                },
            },
        )
        logger.info(f"Connection test result: {result}")
        return True

    except Exception as e:
        logger.error(f"Connection test failed: {e}")
        return False


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    test_gemini_connection()
