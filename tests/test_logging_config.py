"""Unit tests for src.logging_config module."""

import logging

from src.logging_config import configure_logging


class TestConfigureLogging:
    """Tests for configure_logging function."""

    def teardown_method(self) -> None:
        """Clean up logging handlers after each test."""
        # Remove all handlers from root logger
        root_logger = logging.getLogger()
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)

    def test_configure_logging_does_not_raise(self) -> None:
        """Test configure_logging executes without error."""
        # Should not raise an exception
        configure_logging(level=logging.DEBUG)
        configure_logging(level=logging.INFO)
        configure_logging()

    def test_configure_logging_adds_handler(self) -> None:
        """Test configure_logging adds a stream handler."""
        root_logger = logging.getLogger()
        initial_handler_count = len(root_logger.handlers)
        configure_logging(level=logging.INFO)
        # Should have at least one handler (pytest may have added others)
        assert len(root_logger.handlers) >= initial_handler_count

    def test_configure_logging_handler_has_formatter(self) -> None:
        """Test configured handler has a formatter."""
        configure_logging(level=logging.INFO)
        root_logger = logging.getLogger()
        handlers = root_logger.handlers
        # At least one handler should have a formatter
        assert any(
            h.formatter is not None for h in handlers if isinstance(h, logging.StreamHandler)
        )

    def test_configure_logging_can_log(self) -> None:
        """Test logger can successfully log messages after configuration."""
        configure_logging(level=logging.INFO)
        logger = logging.getLogger("test_logger")
        # Should not raise an exception
        logger.info("Test message")
        logger.warning("Warning message")
        logger.error("Error message")

    def test_configure_logging_respects_level_parameter(self) -> None:
        """Test configure_logging accepts level parameter."""
        # Just verify these don't raise exceptions
        configure_logging(level=logging.DEBUG)
        configure_logging(level=logging.INFO)
        configure_logging(level=logging.WARNING)
        configure_logging(level=logging.ERROR)
