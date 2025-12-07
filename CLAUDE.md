This repo is for building an ambient Claude discord bot. The main code is in bot.py. The latest conversation (system prompt, user context, and Claude's tool uses/responses) is saved to latest_conversation.txt.

## If you are asked to run the server...

Before starting the server, check for existing instances:

```
powershell "Get-WmiObject Win32_Process -Filter \"name='python3.13.exe'\" | Select-Object ProcessId, CommandLine | Format-List"
```

Kill any existing bot.py processes before starting a new one:

```
powershell "Stop-Process -Id <PID> -Force"
```

Then use nodemon for auto-restart on file changes:

```
nodemon --exec python bot.py --ext py
```

## API Notes

- `web_search` tool: type `web_search_20250305`, no beta header needed
- `web_fetch` tool: type `web_fetch_20250910`, requires beta header `anthropic-beta: web-fetch-2025-09-10`
