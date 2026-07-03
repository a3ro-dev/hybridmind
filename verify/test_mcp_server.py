"""
Verify MCP server tools mapping.
"""
import asyncio

from mcp_server.main import mcp

def test_mcp_tools_registration():
    # Verify tools are discoverable through the public FastMCP API.
    tools = asyncio.run(mcp.list_tools())
    tool_names = {tool.name for tool in tools}
    assert "remember" in tool_names
    assert "recall" in tool_names
    assert "relate" in tool_names
    assert "forget" in tool_names

def test_mcp_tool_signatures():
    # Assert recall signature takes query and returns results
    tools = asyncio.run(mcp.list_tools())
    recall_tool = next(tool for tool in tools if tool.name == "recall")
    assert "query" in recall_tool.inputSchema["properties"]
