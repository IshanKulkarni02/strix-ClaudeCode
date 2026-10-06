"""``strix mcp-server`` — expose Strix's tools over the MCP stdio transport.

Lets an MCP client (Claude Code, Cursor, Zed, ...) call Strix's existing tools
directly. The client supplies the model and the orchestration loop, so this
mode needs neither ``STRIX_LLM`` nor ``LLM_API_KEY``. Strix contributes the
sandbox container, the proxy, the skill packs, and the reporting tools.

This is additive: ``strix --target ...`` still runs the native agent graph
against a configured LLM, unchanged.

Strix's orchestration and bookkeeping tools (agent graph, ``finish_scan``,
``wait_for_user``, notes, todos, ``think``) are intentionally not exposed — the
MCP client brings its own planning, memory, and control flow.

stdio transport owns stdout, so all logging here goes to stderr.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import logging
import sys
import uuid
from typing import TYPE_CHECKING, Any

# ExecCommandTool / WriteStdinTool are not re-exported from the capabilities
# package, so they are imported from their defining module. Strix pins the
# agents SDK to >=0.19,<0.20 (see pyproject), so this internal path is stable
# for the supported SDK range.
from agents.sandbox.capabilities.shell import (  # type: ignore[attr-defined]
    ExecCommandTool,
    WriteStdinTool,
)
from agents.tool_context import ToolContext
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import TextContent
from mcp.types import Tool as MCPTool

from strix.config import load_settings
from strix.interface.cli_args import get_version
from strix.runtime import session_manager
from strix.tools.coverage.tools import list_coverage, record_coverage, update_coverage
from strix.tools.load_skill.tool import load_skill
from strix.tools.proxy.tools import (
    list_requests,
    list_sitemap,
    repeat_request,
    scope_rules,
    view_request,
    view_sitemap_entry,
)
from strix.tools.reporting.tool import (
    create_dependency_report,
    create_vulnerability_report,
    delete_vulnerability_report,
    get_report,
    list_reports,
    update_vulnerability_report,
)
from strix.tools.threat_model.tools import (
    amend_threat_model,
    get_threat_model,
    save_threat_model,
)
from strix.tools.web_search.tool import web_get_contents, web_search


if TYPE_CHECKING:
    from agents.tool import FunctionTool


logger = logging.getLogger(__name__)

SERVER_NAME = "strix"
_CALL_ID_PREFIX = "mcp"

_INSTRUCTIONS = (
    "Strix tools over MCP. Shell tools (exec_command, write_stdin) run inside "
    "an isolated Strix sandbox container that is started on first use. Proxy "
    "tools expose the HTTP history captured by the sandbox's built-in proxy. "
    "Reporting tools record CVSS-scored findings. load_skill loads Strix's "
    "internal knowledge packs. Only operate against targets you are authorized "
    "to test."
)


def _host_tools() -> list[FunctionTool]:
    """Host-side tools that need no live sandbox session.

    These read their run state from a plain ``ctx.context`` dict and degrade
    gracefully when optional keys (coordinator, caido client) are absent, so
    they are safe to expose with the minimal context this server builds.
    """
    return [
        load_skill,
        web_search,
        web_get_contents,
        list_requests,
        view_request,
        repeat_request,
        list_sitemap,
        view_sitemap_entry,
        scope_rules,
        create_vulnerability_report,
        update_vulnerability_report,
        delete_vulnerability_report,
        create_dependency_report,
        get_report,
        list_reports,
        get_threat_model,
        save_threat_model,
        amend_threat_model,
        list_coverage,
        record_coverage,
        update_coverage,
    ]


# Sandbox tools run commands inside the container. Their input schemas are
# static, so they can be advertised without a live session; only execution
# needs one. apply_patch is a CustomTool (a different invoke contract) and
# view_image returns image bytes that don't fit an MCP text result, so neither
# is exposed here — the shell covers file edits and inspection inside the box.
def _sandbox_tool_specs() -> list[FunctionTool]:
    """Unbound sandbox tools, built only to read their name and schema."""
    return [
        ExecCommandTool(session=None),  # type: ignore[arg-type]  # schema-only, never invoked
        WriteStdinTool(session=None),  # type: ignore[arg-type]  # schema-only, never invoked
    ]


SANDBOX_TOOL_NAMES = frozenset(t.name for t in _sandbox_tool_specs())


class _SandboxTools:
    """Lazily brings up one Strix sandbox session and binds the shell tools to
    it. The session is created on the first call that needs it and reused for
    the life of the server; ``aclose`` tears it down."""

    def __init__(self) -> None:
        self._scan_id = uuid.uuid4().hex[:8]
        self._bundle: dict[str, Any] | None = None
        self._tools: dict[str, FunctionTool] | None = None
        self._lock = asyncio.Lock()

    async def ensure(self) -> dict[str, FunctionTool]:
        async with self._lock:
            if self._tools is not None:
                return self._tools
            self._tools = await self._bring_up()
            return self._tools

    async def _bring_up(self) -> dict[str, FunctionTool]:
        settings = load_settings()
        logger.info("Starting Strix sandbox session %s", self._scan_id)
        self._bundle = await session_manager.create_or_reuse(
            self._scan_id,
            image=settings.runtime.image,
            local_sources=[],
            status_sink=lambda phase: logger.info("sandbox: %s", phase),
        )
        session = self._bundle["session"]

        tools: list[FunctionTool] = [ExecCommandTool(session=session)]
        if session.supports_pty():
            tools.append(WriteStdinTool(session=session))
        logger.info("Sandbox ready; bound %d sandbox tool(s)", len(tools))
        return {t.name: t for t in tools}

    def caido_client(self) -> Any:
        return self._bundle["caido_client"] if self._bundle else None

    async def aclose(self) -> None:
        if self._bundle is None:
            return
        with contextlib.suppress(Exception):
            await session_manager.cleanup(self._scan_id)
        logger.info("Sandbox session %s cleaned up", self._scan_id)


def _run_context(sandbox: _SandboxTools) -> dict[str, Any]:
    """The minimal ``ctx.context`` dict Strix tools read from.

    Only the keys the exposed tools actually consult are populated; the rest of
    Strix's run context (coordinator, agent graph) is absent and the tools fall
    back to their no-coordinator paths.
    """
    return {
        "caido_client": sandbox.caido_client(),
        "agent_id": SERVER_NAME,
        "parent_id": None,
        "interactive": False,
    }


async def _invoke(tool: FunctionTool, arguments: dict[str, Any], sandbox: _SandboxTools) -> str:
    """Call a Strix FunctionTool and return its result as text."""
    ctx: ToolContext[Any] = ToolContext(
        context=_run_context(sandbox),
        tool_name=tool.name,
        tool_call_id=f"{_CALL_ID_PREFIX}_{uuid.uuid4().hex[:12]}",
        tool_arguments=json.dumps(arguments),
    )
    result = await tool.on_invoke_tool(ctx, json.dumps(arguments))
    if isinstance(result, str):
        return result
    return json.dumps(result, default=str)


def _build_server(sandbox: _SandboxTools) -> Any:
    server: Any = Server(SERVER_NAME, version=get_version(), instructions=_INSTRUCTIONS)

    host_tools = {t.name: t for t in _host_tools()}

    def _as_mcp_tool(tool: FunctionTool) -> MCPTool:
        return MCPTool(
            name=tool.name,
            description=(tool.description or "").strip() or tool.name,
            inputSchema=tool.params_json_schema or {"type": "object", "properties": {}},
        )

    @server.list_tools()  # type: ignore[untyped-decorator]  # mcp SDK decorators are untyped
    async def list_tools() -> list[MCPTool]:
        # Sandbox tools are advertised from their static schemas, so listing
        # never starts the container; the session is created on first call.
        return [_as_mcp_tool(t) for t in (*host_tools.values(), *_sandbox_tool_specs())]

    @server.call_tool()  # type: ignore[untyped-decorator]  # mcp SDK decorators are untyped
    async def call_tool(name: str, arguments: dict[str, Any]) -> list[TextContent]:
        tool: FunctionTool | None
        if name in host_tools:
            tool = host_tools[name]
        elif name in SANDBOX_TOOL_NAMES:
            tool = (await sandbox.ensure()).get(name)
            if tool is None:
                # Advertised, but this session doesn't provide it (e.g. no PTY).
                raise ValueError(f"tool not available in this sandbox session: {name}")
        else:
            raise ValueError(f"unknown tool: {name}")
        text = await _invoke(tool, arguments or {}, sandbox)
        return [TextContent(type="text", text=text)]

    return server


async def _serve() -> None:
    sandbox = _SandboxTools()
    server = _build_server(sandbox)
    init_options = server.create_initialization_options()
    try:
        async with stdio_server() as (read_stream, write_stream):
            await server.run(read_stream, write_stream, init_options)
    finally:
        await sandbox.aclose()


def run_mcp_server(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        prog="strix mcp-server",
        description="Serve Strix's tools to an MCP client over stdio.",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Log at DEBUG level to stderr.",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        stream=sys.stderr,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    try:
        asyncio.run(_serve())
    except KeyboardInterrupt:
        return 0
    return 0
