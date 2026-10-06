"""Tests for ``strix mcp-server`` — the MCP stdio bridge.

These cover the parts that need no Docker sandbox: the host-tool registry, the
MCP tool advertisement, and the invoke path that calls a Strix ``FunctionTool``
through a minimal run context. Sandbox bring-up is exercised separately and is
not required here.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest
from agents.tool import FunctionTool
from mcp.shared.memory import create_connected_server_and_client_session

from strix.interface import mcp_server
from strix.tools.load_skill.tool import load_skill


if TYPE_CHECKING:
    from pathlib import Path

    from agents.tool_context import ToolContext


def test_host_tools_are_function_tools_with_schemas() -> None:
    tools = mcp_server._host_tools()
    assert tools, "expected at least one host tool"
    for tool in tools:
        assert isinstance(tool, FunctionTool)
        assert tool.name
        assert isinstance(tool.params_json_schema, dict)


def test_host_tools_have_unique_names() -> None:
    names = [t.name for t in mcp_server._host_tools()]
    assert len(names) == len(set(names))


def test_orchestration_tools_are_not_exposed() -> None:
    names = {t.name for t in mcp_server._host_tools()}
    # The MCP client owns planning, control flow, and memory; Strix's
    # equivalents must not be advertised or they fight the client.
    for banned in ("create_agent", "finish_scan", "wait_for_user", "think", "create_todo"):
        assert banned not in names


def test_run_context_has_no_coordinator() -> None:
    # The exposed tools must degrade gracefully with no agent graph present.
    ctx = mcp_server._run_context(mcp_server._SandboxTools())
    assert ctx["agent_id"] == mcp_server.SERVER_NAME
    assert "coordinator" not in ctx
    assert ctx["interactive"] is False


@pytest.mark.asyncio
async def test_invoke_returns_text_for_bad_input() -> None:
    # load_skill with an invalid argument should come back as a graceful string,
    # never raise — the invoke path must always yield text for MCP.
    sandbox = mcp_server._SandboxTools()
    result = await mcp_server._invoke(load_skill, {"skills": 12345}, sandbox)
    assert isinstance(result, str)
    assert result


@pytest.mark.asyncio
async def test_invoke_serializes_non_string_results() -> None:
    # A tool returning a non-string must be JSON-encoded, not stringified ad hoc.
    async def _fake_invoke(_ctx: ToolContext[object], _raw: str) -> dict[str, int]:
        return {"ok": 1}

    fake = FunctionTool(
        name="fake",
        description="fake",
        params_json_schema={"type": "object", "properties": {}},
        on_invoke_tool=_fake_invoke,
    )
    sandbox = mcp_server._SandboxTools()
    result = await mcp_server._invoke(fake, {}, sandbox)
    assert json.loads(result) == {"ok": 1}


def test_sandbox_specs_readable_without_a_session() -> None:
    # Advertising sandbox tools must not require a live container.
    specs = mcp_server._sandbox_tool_specs()
    names = {t.name for t in specs}
    assert names == set(mcp_server.SANDBOX_TOOL_NAMES)
    assert names == {"exec_command", "write_stdin"}
    for tool in specs:
        assert isinstance(tool.params_json_schema, dict)


@pytest.mark.asyncio
async def test_list_tools_handshake_does_not_start_sandbox() -> None:
    # A client lists tools on connect; that must complete over a real MCP
    # session without ever bringing up the Docker sandbox.
    class _NoBootSandbox(mcp_server._SandboxTools):
        async def ensure(self) -> dict[str, FunctionTool]:
            raise AssertionError("list_tools must not bring up the sandbox")

    server = mcp_server._build_server(_NoBootSandbox())
    async with create_connected_server_and_client_session(server) as client:
        listed = await client.list_tools()

    advertised = {t.name for t in listed.tools}
    assert "exec_command" in advertised  # sandbox tool, advertised statically
    assert "write_stdin" in advertised
    assert "load_skill" in advertised  # host tool
    # Orchestration tools stay hidden end-to-end.
    assert "finish_scan" not in advertised


@pytest.mark.asyncio
async def test_call_host_tool_over_session_returns_text() -> None:
    # A full client->server call of a host tool (no sandbox) returns text
    # content and does not raise, even on invalid input.
    server = mcp_server._build_server(mcp_server._SandboxTools())
    async with create_connected_server_and_client_session(server) as client:
        result = await client.call_tool("load_skill", {"skills": 123})
    assert result.content
    assert result.content[0].type == "text"
    assert result.content[0].text


@pytest.mark.asyncio
async def test_call_unknown_tool_errors() -> None:
    server = mcp_server._build_server(mcp_server._SandboxTools())
    async with create_connected_server_and_client_session(server) as client:
        result = await client.call_tool("no_such_tool", {})
    assert result.isError


def test_build_server_advertises_host_tools() -> None:
    server = mcp_server._build_server(mcp_server._SandboxTools())
    assert server.name == mcp_server.SERVER_NAME


def test_run_mcp_server_rejects_unknown_flag() -> None:
    with pytest.raises(SystemExit):
        mcp_server.run_mcp_server(["--definitely-not-a-flag"])


def test_build_local_sources_maps_dirs_to_mounts(tmp_path: Path) -> None:
    repo = tmp_path / "my-repo"
    repo.mkdir()
    sources = mcp_server._build_local_sources([str(repo)])
    assert len(sources) == 1
    assert sources[0]["source_path"] == str(repo)
    assert sources[0]["workspace_subdir"] == "my-repo"


def test_build_local_sources_rejects_missing_dir(tmp_path: Path) -> None:
    missing = tmp_path / "nope"
    with pytest.raises(ValueError, match="not a directory"):
        mcp_server._build_local_sources([str(missing)])


def test_mounted_repo_appears_in_context_and_instructions(tmp_path: Path) -> None:
    repo = tmp_path / "target-app"
    repo.mkdir()
    sources = mcp_server._build_local_sources([str(repo)])
    sandbox = mcp_server._SandboxTools(local_sources=sources)
    assert sandbox.workspace_paths == ["/workspace/target-app"]
    # Tools scope to the mounted path, and the client is told where it is.
    assert mcp_server._run_context(sandbox)["scan_targets"] == ["/workspace/target-app"]
    assert "/workspace/target-app" in mcp_server._server_instructions(sandbox)


def test_no_mount_leaves_instructions_unchanged() -> None:
    sandbox = mcp_server._SandboxTools()
    assert sandbox.workspace_paths == []
    assert mcp_server._server_instructions(sandbox) == mcp_server._INSTRUCTIONS
