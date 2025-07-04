import spacy
from langchain_core.tools import tool
import os
import subprocess
import sys
import warnings

# Global variable to hold the loaded model
nlp = None
MODEL_NAME = "en_core_web_sm"
_model_loaded = False  # Track if loading was successful


def _ensure_model_loaded():
    """Loads the spaCy model, downloading if necessary. Returns True if successful."""
    global nlp, _model_loaded
    if _model_loaded:
        return True  # Already loaded successfully

    if nlp is not None:  # Check if already loaded (might happen in some contexts)
        _model_loaded = True
        return True

    try:
        # Check if spacy is installed
        import spacy
    except ImportError:
        warnings.warn(
            f"ImportError: spaCy library not found. Please install it (e.g., pip install spacy). NER tool will not function."
        )
        nlp = None
        _model_loaded = False
        return False

    try:
        nlp = spacy.load(MODEL_NAME)
        print(f"Successfully loaded spaCy model '{MODEL_NAME}'.")
        _model_loaded = True
        return True
    except OSError:
        warnings.warn(
            f"spaCy model '{MODEL_NAME}' not found. Attempting to download..."
        )
        try:
            # Use subprocess to run the download command
            command = [sys.executable, "-m", "spacy", "download", MODEL_NAME]
            print(f"Running command: {' '.join(command)}")
            # Use check_output to capture stdout/stderr for better debugging if needed
            subprocess.check_output(command, stderr=subprocess.STDOUT)
            print(
                f"Successfully downloaded spaCy model '{MODEL_NAME}'. Attempting to load..."
            )
            # Need to reload spacy or potentially restart the process
            # for the new package to be recognized reliably.
            # For now, try loading again directly.
            nlp = spacy.load(MODEL_NAME)
            print(f"Successfully loaded spaCy model '{MODEL_NAME}' after download.")
            _model_loaded = True
            return True
        except subprocess.CalledProcessError as e:
            warnings.warn(
                f"Error downloading spaCy model '{MODEL_NAME}' using subprocess: {e.output.decode()}"
            )
            nlp = None
            _model_loaded = False
            return False
        except Exception as e:
            warnings.warn(
                f"Error loading spaCy model '{MODEL_NAME}' after download attempt: {e}"
            )
            nlp = None
            _model_loaded = False
            return False
    except Exception as e:
        warnings.warn(
            f"An unexpected error occurred loading the spaCy model '{MODEL_NAME}': {e}"
        )
        nlp = None
        _model_loaded = False
        return False


# Attempt to load the model when the module is imported.
# We store the result but don't block import if it fails.
_ensure_model_loaded()


@tool
def ner_tool(text: str) -> dict:
    """
    Performs Named Entity Recognition (NER) on the input text using the spaCy 'en_core_web_sm' model.

    Requires the 'spacy' library and the 'en_core_web_sm' model.
    It will attempt to download the model if it's missing.

    Args:
        text: The input string to process.

    Returns:
        A dictionary containing a list of identified entities, where each
        entity is a dictionary with 'text' and 'type' keys.
        Example: {'entities': [{'text': 'Apple', 'type': 'ORG'}, {'text': 'Cupertino', 'type': 'LOC'}]}
        Returns {'error': 'spaCy model 'en_core_web_sm' not available or failed to load.'} if the model isn't loaded.
        Returns {'entities': []} if no entities are found or input is empty/invalid.
    """
    if not _model_loaded or nlp is None:
        # Maybe try one more time? Or rely on the initial load attempt.
        # For simplicity, we rely on the initial load attempt status.
        return {
            "error": f"spaCy model '{MODEL_NAME}' not available or failed to load. Please ensure 'spacy' is installed and the model can be downloaded/loaded."
        }

    if not isinstance(text, str) or not text:
        return {"entities": []}

    try:
        doc = nlp(text)
        entities = [{"text": ent.text, "type": ent.label_} for ent in doc.ents]
        return {"entities": entities}
    except Exception as e:
        # Catch potential errors during processing
        print(f"Error processing text with spaCy: {e}")  # Log error
        return {"error": f"Failed to process text with spaCy: {e}"}
