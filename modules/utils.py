# modules/utils.py
import logging
import streamlit as st # Needed for st.session_state

def log_error(context, error):
    """Logs an error message with context and adds to session state log."""
    try:
        estr = str(error)
    except Exception:
        estr = "Unstringifiable Error Object"
    msg = f"ERROR [{context}]: {estr}"
    logging.error(msg)
    # Use setdefault to initialize if 'error_log' doesn't exist
    # Check if session state is available (might not be if run outside Streamlit context)
    try:
        if st.runtime.exists(): # Check if running in Streamlit session
            st.session_state.setdefault("error_log", []).append(msg)
    except Exception:
        # Handle cases where session state might not be accessible (e.g., testing)
        logging.warning("Could not append error to Streamlit session state.")

