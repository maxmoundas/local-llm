"""A minimal MCP server for connector tests: `python mcp_test_server.py` (stdio)."""
try:  # mcp >= 2
    from mcp.server.mcpserver import MCPServer as Server
except ImportError:  # mcp 1.x
    from mcp.server.fastmcp import FastMCP as Server

mcp = Server("test")


@mcp.tool()
def add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b


@mcp.tool()
def send_message(to: str, body: str) -> str:
    """Send a message to someone."""
    return f"sent to {to}"


if __name__ == "__main__":
    mcp.run()
