# Connect VeraMem to your AI app (MCP)

Run `veramem mcp-config` to print the exact JSON for your machine, then paste it into the
client's config file.

| Client | Where it goes |
|---|---|
| Claude Desktop | Settings → Developer → Edit Config (`claude_desktop_config.json`), under `mcpServers` |
| Claude Code | `claude mcp add veramem -- veramem mcp` |
| Cursor | `~/.cursor/mcp.json` |
| VS Code | `.vscode/mcp.json` |

```json
{
  "mcpServers": {
    "veramem": {
      "command": "veramem",
      "args": ["mcp"],
      "env": { "VERAMEM_USER": "your-name" }
    }
  }
}
```

The server's user is fixed by `VERAMEM_USER`. The model cannot pick another user's id.
