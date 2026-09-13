"""Web dashboard: FastAPI backend + phone-first static UI."""

from .server import AccessControl, create_app, serve, serve_in_thread

__all__ = ["AccessControl", "create_app", "serve", "serve_in_thread"]
