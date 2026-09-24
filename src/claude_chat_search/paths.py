import os
from pathlib import Path

# CLAUDE_CHAT_SEARCH_HOME relocates everything the tool writes (index, queue,
# logs, exclusion list, backups); CHAT_SEARCH_DB_PATH still overrides just the DB.
DATA_DIR = Path(
    os.environ.get("CLAUDE_CHAT_SEARCH_HOME") or Path.home() / ".claude-chat-search"
).expanduser()
