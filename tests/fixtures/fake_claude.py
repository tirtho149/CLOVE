#!/usr/bin/env python3
"""Stand-in for the `claude` CLI used by tests (no network, no account).

Reads one stream-json user message, answers the first lettered option, and emits
stream-json events shaped like Claude Code's. FAKE_CLAUDE_MODE selects misbehaviour:
  ok (default) | read_other (Reads a file other than the query) | error | tool_inline
"""
import json
import os
import re
import sys

if "--version" in sys.argv:
    print("9.9.9 (Fake Claude)")
    sys.exit(0)
mode = os.environ.get("FAKE_CLAUDE_MODE", "ok")
log = os.environ.get("FAKE_CLAUDE_LOG")
msg = json.loads(sys.stdin.readline())
content = msg["message"]["content"]
kinds = [c["type"] for c in content]
text = "".join(c.get("text", "") for c in content if c["type"] == "text")
tools = sys.argv[sys.argv.index("--tools") + 1]
if log:
    with open(log, "a") as f:
        f.write(json.dumps({"argv": sys.argv[1:], "kinds": kinds, "env_claude": sorted(k for k in os.environ if k.startswith("CLAUDE")),
                            "has_api_key": "ANTHROPIC_API_KEY" in os.environ,
                            "env_all": sorted(k for k in os.environ if "CACHING" in k), "cwd": os.getcwd()}) + "\n")
if mode == "cap":
    print(json.dumps({"type": "result", "is_error": True, "total_cost_usd": 0.05,
                      "result": "API Error: Claude's response exceeded the 96 output token maximum.",
                      "usage": {"input_tokens": 16000, "output_tokens": 96}}))
    sys.exit(1)
if mode == "error":
    print(json.dumps({"type": "result", "is_error": True, "result": "overloaded_error"}))
    sys.exit(1)
events = []
if tools == "Read":
    path = re.search(r"`([^`]+)`", text).group(1)
    if mode == "read_other":
        path = "/tmp/some_reference_image.jpg"
    events.append({"type": "assistant", "message": {"content": [{"type": "tool_use", "name": "Read", "input": {"file_path": path}}]}})
if mode == "tool_inline":
    events.append({"type": "assistant", "message": {"content": [{"type": "tool_use", "name": "Glob", "input": {"pattern": "*"}}]}})
opts = re.findall(r"^([A-Z]{1,2})\) ", text, re.M)
if opts:
    ans = json.dumps({"answer": opts[0]})
else:
    cls = re.findall(r"^- (\S+)$", text, re.M)
    ans = "```json\n" + json.dumps({"prediction": cls[0] if cls else "UNKNOWN", "confidence": 0.5, "reasoning": "x"}) + "\n```"
events.append({"type": "assistant", "message": {"content": [{"type": "text", "text": ans}]}})
events.append({"type": "result", "is_error": False, "result": ans, "num_turns": 1 + (tools == "Read"),
               "duration_ms": 5, "total_cost_usd": 0.001,
               "usage": {"input_tokens": 3, "cache_creation_input_tokens": 1500, "cache_read_input_tokens": 0,
                         "output_tokens": 9},
               "modelUsage": {sys.argv[sys.argv.index("--model") + 1]: {
                   "inputTokens": 3, "cacheCreationInputTokens": 1500, "cacheReadInputTokens": 0,
                   "outputTokens": 9, "costUSD": 0.001}}})
for e in events:
    print(json.dumps(e))
