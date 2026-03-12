import ssl

# Save the original method
_original_load_default_certs = ssl.SSLContext.load_default_certs

def _patched_load_default_certs(self, purpose=ssl.Purpose.SERVER_AUTH):
    try:
        _original_load_default_certs(self, purpose)
    except ssl.SSLError:
        # Windows store has a bad cert — fall back to certifi
        import certifi
        self.load_verify_locations(certifi.where())

ssl.SSLContext.load_default_certs = _patched_load_default_certs

# Now run Streamlit normally
from streamlit.web.cli import main
main()