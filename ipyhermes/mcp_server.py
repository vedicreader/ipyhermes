"MCP server exposing ipyhermes/karma tools via Model Context Protocol."
import json

__all__ = ['create_server', 'main']


def _get_tools():
    "Return tool definitions for the MCP server."
    return [
        dict(name="dev_context", description="Get development context for a task",
             inputSchema=dict(type="object", properties=dict(
                 query=dict(type="string", description="The task or query"),
                 repo_root=dict(type="string", description="Repository root path", default="."),
             ), required=["query"])),
        dict(name="search_code", description="Search indexed code",
             inputSchema=dict(type="object", properties=dict(
                 query=dict(type="string", description="Search query"),
                 repo_root=dict(type="string", description="Repository root path", default="."),
                 k=dict(type="integer", description="Number of results", default=5),
             ), required=["query"])),
        dict(name="index_repo", description="Index a repository for code search",
             inputSchema=dict(type="object", properties=dict(
                 repo_root=dict(type="string", description="Repository root path", default="."),
                 force=dict(type="boolean", description="Force re-index", default=False),
             ))),
        dict(name="add_practice", description="Add a practice/convention to memory",
             inputSchema=dict(type="object", properties=dict(
                 rule=dict(type="string", description="The practice rule"),
                 severity=dict(type="string", description="must|should|avoid|note", default="should"),
                 tags=dict(type="string", description="Comma-separated tags", default=""),
                 repo_root=dict(type="string", description="Repository root", default="."),
             ), required=["rule"])),
        dict(name="query_practices", description="Query practice memory",
             inputSchema=dict(type="object", properties=dict(
                 query=dict(type="string", description="Search query"),
                 repo_root=dict(type="string", description="Repository root", default="."),
                 k=dict(type="integer", description="Number of results", default=5),
             ), required=["query"])),
        dict(name="log_decision", description="Log an architectural decision",
             inputSchema=dict(type="object", properties=dict(
                 content=dict(type="string", description="Decision description"),
                 source=dict(type="string", description="human or llm", default="human"),
             ), required=["content"])),
        dict(name="search_decisions", description="Search decision log",
             inputSchema=dict(type="object", properties=dict(
                 query=dict(type="string", description="Search query"),
                 k=dict(type="integer", description="Number of results", default=5),
             ), required=["query"])),
        dict(name="search_tool_calls", description="Search tool call history",
             inputSchema=dict(type="object", properties=dict(
                 query=dict(type="string", description="Search query"),
                 k=dict(type="integer", description="Number of results", default=10),
             ), required=["query"])),
    ]


def _call_tool(name: str, arguments: dict) -> str:
    "Dispatch a tool call to the appropriate karma/ipyhermes function."
    try:
        if name == "dev_context":
            from karma.skill import dev_context
            return str(dev_context(arguments.get("query", ""), arguments.get("repo_root", ".")))
        elif name == "search_code":
            from karma.skill import search_code
            return json.dumps(search_code(arguments.get("query", ""),
                                          repo_root=arguments.get("repo_root", "."),
                                          k=arguments.get("k", 5)))
        elif name == "index_repo":
            from karma.skill import index_repo
            return json.dumps(index_repo(arguments.get("repo_root", "."),
                                         force=arguments.get("force", False)))
        elif name == "add_practice":
            from karma.skill import add_practice
            return str(add_practice(arguments["rule"],
                                    severity=arguments.get("severity", "should"),
                                    tags=arguments.get("tags", ""),
                                    repo_root=arguments.get("repo_root", ".")))
        elif name == "query_practices":
            from karma.skill import query_practices
            return json.dumps(query_practices(arguments.get("query", ""),
                                              repo_root=arguments.get("repo_root", "."),
                                              k=arguments.get("k", 5)))
        elif name == "log_decision":
            from karma.skill import log_decision
            return str(log_decision(arguments["content"],
                                    source=arguments.get("source", "human")))
        elif name == "search_decisions":
            from karma.skill import search_decisions
            return json.dumps(search_decisions(arguments.get("query", ""),
                                               k=arguments.get("k", 5)))
        elif name == "search_tool_calls":
            from ipyhermes.toollog import ToolCallLog
            log = ToolCallLog()
            return json.dumps(log.search(arguments.get("query", ""),
                                         k=arguments.get("k", 10)))
        else:
            return json.dumps(dict(error=f"Unknown tool: {name}"))
    except ImportError as e:
        return json.dumps(dict(error=f"Dependency not available: {e}"))
    except Exception as e:
        return json.dumps(dict(error=str(e)))


def create_server():
    "Create and configure the MCP server."
    try:
        from mcp.server import Server
        from mcp.types import TextContent, Tool
    except ImportError:
        raise ImportError("mcp package is required for MCP server. Install with: pip install mcp")

    server = Server("ipyhermes")

    @server.list_tools()
    async def list_tools():
        return [Tool(**t) for t in _get_tools()]

    @server.call_tool()
    async def call_tool(name: str, arguments: dict):
        result = _call_tool(name, arguments or {})
        return [TextContent(type="text", text=result)]

    return server


def main():
    "Run the MCP server via stdio transport."
    import asyncio

    async def _run():
        try:
            from mcp.server.stdio import stdio_server
        except ImportError:
            print("mcp package is required. Install with: pip install mcp", file=__import__('sys').stderr)
            raise SystemExit(1)

        server = create_server()
        async with stdio_server() as (read_stream, write_stream):
            await server.run(read_stream, write_stream, server.create_initialization_options())

    asyncio.run(_run())


if __name__ == "__main__":
    main()
