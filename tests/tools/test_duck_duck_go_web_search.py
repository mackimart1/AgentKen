import unittest
from unittest.mock import patch, MagicMock

# Adjust import path if necessary
try:
    from tools.duck_duck_go_web_search import duck_duck_go_web_search
except ImportError:
    import sys
    import os

    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
    from tools.duck_duck_go_web_search import duck_duck_go_web_search


class TestDuckDuckGoWebSearch(unittest.TestCase):

    # Patch the DuckDuckGoSearchResults class within the tool's module
    @patch("tools.duck_duck_go_web_search.DuckDuckGoSearchResults")
    def test_search_success(self, MockDuckDuckGoSearchResults):
        """Tests successful search by mocking DuckDuckGoSearchResults."""

        # Configure the mock search instance
        mock_search_instance = MagicMock()
        mock_search_instance.invoke.return_value = "[{'title': 'Result 1', 'link': 'http://example.com/1', 'snippet': 'Snippet 1...'}]"

        # Make the patched class return our mock instance when instantiated
        MockDuckDuckGoSearchResults.return_value = mock_search_instance

        query = "test query"
        result = duck_duck_go_web_search.invoke({"query": query})

        # Assert DuckDuckGoSearchResults was instantiated (implicitly)
        MockDuckDuckGoSearchResults.assert_called_once()

        # Assert the invoke method was called on the instance with the query
        mock_search_instance.invoke.assert_called_once_with(query)

        # Assert the tool returned the expected result from the mock
        self.assertEqual(
            result,
            "[{'title': 'Result 1', 'link': 'http://example.com/1', 'snippet': 'Snippet 1...'}]",
        )

    @patch("tools.duck_duck_go_web_search.DuckDuckGoSearchResults")
    def test_search_error(self, MockDuckDuckGoSearchResults):
        """Tests handling when the underlying search tool raises an exception."""

        # Configure the mock search instance to raise an error
        mock_search_instance = MagicMock()
        mock_search_instance.invoke.side_effect = Exception("Search API error")
        MockDuckDuckGoSearchResults.return_value = mock_search_instance

        query = "error query"

        # Expect the tool to raise the exception from the underlying library
        with self.assertRaisesRegex(Exception, "Search API error"):
            duck_duck_go_web_search.invoke({"query": query})

        # Assert DuckDuckGoSearchResults was instantiated
        MockDuckDuckGoSearchResults.assert_called_once()
        # Assert the invoke method was called
        mock_search_instance.invoke.assert_called_once_with(query)


if __name__ == "__main__":
    unittest.main()
