"""
Verify MCP server tools mapping.
"""
from mcp_server.main import mcp

def test_mcp_tools_registration():
    # Verify tools exist on FastMCP instance
    tools = mcp._tools
    assert "remember" in tools
    assert "recall" in tools
    assert "relate" in tools
    assert "forget" in tools

def test_mcp_tool_signatures():
    # Assert recall signature takes query and returns results
    recall_tool = mcp._tools["recall"]
    assert "query" in recall_tool.arguments
