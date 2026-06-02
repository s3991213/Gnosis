"""
FastMCP server entry point for Gnosis.

Instantiates the MCP server, wires up the tool implementations from
``tools.py``, and exposes the server via stdio transport (for Claude Desktop)
or SSE/HTTP transport (for OpenWebUI and other network clients).

Run via::

    gnosis serve
    # or directly:
    python -m gnosis.mcp_server.server
"""

from __future__ import annotations


def create_server():
    """
    Build and return the configured FastMCP application.

    Registers all tools defined in ``gnosis.mcp_server.tools``.
    """
    ...


def run(transport: str = "stdio", host: str = "0.0.0.0", port: int = 8000) -> None:
    """
    Start the MCP server.

    Parameters
    ----------
    transport:
        ``"stdio"`` for Claude Desktop integration, ``"sse"`` for HTTP.
    host:
        Bind address (only relevant for SSE transport).
    port:
        Listen port (only relevant for SSE transport).
    """
    ...


if __name__ == "__main__":
    run()
