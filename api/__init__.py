"""
app/__init__.py - Package initialization
"""
from .entrypoint import app
from . import routes

# Include routes
app.include_router(routes.router)