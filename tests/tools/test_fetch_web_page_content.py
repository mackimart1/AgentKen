import unittest
from unittest.mock import patch, MagicMock

# Adjust import path if necessary
try:
    from tools.fetch_web_page_content import fetch_web_page_content
except ImportError:
    import sys
    import os

    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
    from tools.fetch_web_page_content import fetch_web_page_content


# Mock the Document class that SeleniumURLLoader returns
class MockDocument:
    def __init__(self, page_content, metadata):
        self.page_content = page_content
        self.metadata = metadata

    def __repr__(self):
        return f"MockDocument(page_content='{self.page_content}', metadata={self.metadata})"


class TestFetchWebPageContent(unittest.TestCase):

    # Patch the SeleniumURLLoader within the tool's module
    @patch("tools.fetch_web_page_content.SeleniumURLLoader")
    def test_fetch_content_success(self, MockSeleniumURLLoader):
        """Tests successful content fetching by mocking SeleniumURLLoader."""

        # Configure the mock loader instance
        mock_loader_instance = MagicMock()
        # Simulate the loader returning a list with one Document object
        mock_page = MockDocument(
            page_content="Rendered page content.",
            metadata={"source": "http://example.com"},
        )
        mock_loader_instance.load.return_value = [mock_page]

        # Make the patched class return our mock instance when instantiated
        MockSeleniumURLLoader.return_value = mock_loader_instance

        url = "http://example.com"
        result = fetch_web_page_content.invoke({"url": url})

        # Assert SeleniumURLLoader was instantiated correctly
        MockSeleniumURLLoader.assert_called_once_with(
            urls=[url],
            executable_path="/usr/bin/chromedriver",
            arguments=[
                "--headless",
                "--disable-gpu",
                "--no-sandbox",
                "--disable-dev-shm-usage",
            ],
        )

        # Assert the load method was called on the instance
        mock_loader_instance.load.assert_called_once()

        # Assert the tool returned the expected Document object (or our mock equivalent)
        self.assertIsInstance(result, MockDocument)
        self.assertEqual(result.page_content, "Rendered page content.")
        self.assertEqual(result.metadata["source"], url)

    @patch("tools.fetch_web_page_content.SeleniumURLLoader")
    def test_fetch_content_loader_error(self, MockSeleniumURLLoader):
        """Tests handling when the Selenium loader raises an exception."""

        # Configure the mock loader instance to raise an error on load()
        mock_loader_instance = MagicMock()
        mock_loader_instance.load.side_effect = Exception("Selenium failed to load URL")
        MockSeleniumURLLoader.return_value = mock_loader_instance

        url = "http://invalid-url.fail"

        # Expect the tool to raise the exception from the loader
        with self.assertRaisesRegex(Exception, "Selenium failed to load URL"):
            fetch_web_page_content.invoke({"url": url})

        # Assert SeleniumURLLoader was instantiated
        MockSeleniumURLLoader.assert_called_once()
        # Assert the load method was called
        mock_loader_instance.load.assert_called_once()


if __name__ == "__main__":
    unittest.main()
