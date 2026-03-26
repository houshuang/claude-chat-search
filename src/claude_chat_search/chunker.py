import tiktoken

from .parser import extract_text_content, extract_tool_summary, group_assistant_messages

_encoder = None

MAX_CHUNK_TOKENS = 600
MIN_CHUNK_TOKENS = 150
MERGE_TARGET_TOKENS = 200
OVERLAP_RATIO = 0.25


def get_encoder():
    global _encoder
    if _encoder is None:
        _encoder = tiktoken.get_encoding("cl100k_base")
    return _encoder


def count_tokens(text: str) -> int:
    return len(get_encoder().encode(text))


def split_text_at_paragraphs(text: str, max_tokens: int, overlap_ratio: float = OVERLAP_RATIO) -> list[str]:
    """Split text at paragraph boundaries with overlap."""
    paragraphs = text.split("\n\n")
    chunks = []
    current_parts = []
    current_tokens = 0

    for para in paragraphs:
        para_tokens = count_tokens(para)
        if current_tokens + para_tokens > max_tokens and current_parts:
            chunks.append("\n\n".join(current_parts))
            # Keep last ~25% of paragraphs as overlap
            overlap_target = int(current_tokens * overlap_ratio)
            overlap_parts = []
            overlap_tokens = 0
            for p in reversed(current_parts):
                pt = count_tokens(p)
                if overlap_tokens + pt > overlap_target:
                    break
                overlap_parts.insert(0, p)
                overlap_tokens += pt
            current_parts = overlap_parts
            current_tokens = overlap_tokens

        current_parts.append(para)
        current_tokens += para_tokens

    if current_parts:
        chunks.append("\n\n".join(current_parts))

    return chunks


def _collect_turns(messages: list[dict]) -> list[dict]:
    """Collect user→assistant turn pairs with tool summaries."""
    turns = []
    i = 0
    while i < len(messages):
        msg = messages[i]
        if msg.get("type") == "user":
            user_text = extract_text_content(msg)
            timestamp = msg.get("timestamp")

            # Collect following assistant messages and their tool summaries
            assistant_texts = []
            tool_summaries = []
            j = i + 1
            while j < len(messages) and messages[j].get("type") == "assistant":
                t = extract_text_content(messages[j])
                if t.strip():
                    assistant_texts.append(t)
                ts = extract_tool_summary(messages[j])
                if ts:
                    tool_summaries.append(ts)
                j += 1

            assistant_text = "\n".join(assistant_texts)
            tool_summary = "\n".join(tool_summaries) if tool_summaries else ""

            if user_text.strip() or assistant_text.strip() or tool_summary:
                # Skip noise: no user content and assistant is trivially short
                if not user_text.strip() and len(assistant_text) < 100:
                    i = j
                    continue
                turns.append({
                    "user_text": user_text.strip(),
                    "assistant_text": assistant_text.strip(),
                    "tool_summary": tool_summary,
                    "timestamp": timestamp,
                })
            i = j
        else:
            i += 1

    return turns


def _build_combined(user_text: str, assistant_text: str, tool_summary: str) -> str:
    """Build combined text for a chunk, appending tool summary if present."""
    combined = f"User: {user_text}\n\nAssistant: {assistant_text}"
    if tool_summary:
        combined += f"\n{tool_summary}"
    return combined


_TEMPLATE_PREFIXES = (
    "Your task is to create a detailed summary",
)


def _is_prompt_template(text: str) -> bool:
    """Detect compact/system prompt templates that add noise to search."""
    stripped = text.strip()
    return any(stripped.startswith(p) for p in _TEMPLATE_PREFIXES) if stripped else False


def create_chunks(session_data: dict) -> list[dict]:
    """Create searchable chunks from parsed session data.

    Groups user→assistant turn pairs, enriches with tool context summaries,
    and merges consecutive tiny turns into multi-turn chunks.
    """
    messages = group_assistant_messages(session_data["messages"])
    session_id = session_data["session_id"]
    turns = _collect_turns(messages)

    # First pass: merge consecutive tiny turns
    merged_turns = _merge_tiny_turns(turns)

    # Second pass: create chunks (splitting oversized ones)
    chunks = []
    for turn_num, turn in enumerate(merged_turns):
        if _is_prompt_template(turn["user_text"]):
            continue
        combined = _build_combined(turn["user_text"], turn["assistant_text"], turn["tool_summary"])
        token_count = count_tokens(combined)

        if token_count <= MAX_CHUNK_TOKENS:
            if token_count < MIN_CHUNK_TOKENS and not turn["user_text"].strip():
                # Skip truly empty chunks (no user text, almost no content)
                continue
            chunks.append({
                "session_id": session_id,
                "user_content": turn["user_text"],
                "assistant_content": turn["assistant_text"],
                "combined_text": combined,
                "timestamp": turn["timestamp"],
                "turn_number": turn["turn_number"],
                "token_estimate": token_count,
            })
        else:
            # Split the assistant response
            user_prefix = f"User: {turn['user_text']}\n\n"
            user_tokens = count_tokens(user_prefix)
            available = MAX_CHUNK_TOKENS - user_tokens
            if available < 100:
                available = MAX_CHUNK_TOKENS

            # Include tool summary in the first split part
            text_to_split = turn["assistant_text"]
            if turn["tool_summary"]:
                text_to_split += f"\n{turn['tool_summary']}"

            parts = split_text_at_paragraphs(text_to_split, available)
            for part in parts:
                part_combined = f"User: {turn['user_text']}\n\nAssistant: {part}"
                chunks.append({
                    "session_id": session_id,
                    "user_content": turn["user_text"],
                    "assistant_content": part,
                    "combined_text": part_combined,
                    "timestamp": turn["timestamp"],
                    "turn_number": turn["turn_number"],
                    "token_estimate": count_tokens(part_combined),
                })

    return chunks


def _merge_tiny_turns(turns: list[dict]) -> list[dict]:
    """Merge consecutive small turns into multi-turn chunks.

    Turns under MERGE_TARGET_TOKENS are merged with adjacent turns until
    hitting MAX_CHUNK_TOKENS. This ensures most chunks reach the 200-500
    token sweet spot for embedding quality.
    """
    if not turns:
        return []

    merged = []
    pending = None

    for i, turn in enumerate(turns):
        combined = _build_combined(turn["user_text"], turn["assistant_text"], turn["tool_summary"])
        tokens = count_tokens(combined)

        if pending is None:
            if tokens < MERGE_TARGET_TOKENS:
                # Start accumulating — too small for good embeddings
                pending = {
                    "user_text": turn["user_text"],
                    "assistant_text": turn["assistant_text"],
                    "tool_summary": turn["tool_summary"],
                    "timestamp": turn["timestamp"],
                    "turn_number": i,
                    "tokens": tokens,
                }
            else:
                turn["turn_number"] = i
                merged.append(turn)
        else:
            # We have a pending small turn — try to merge
            merged_user = pending["user_text"]
            if turn["user_text"].strip():
                merged_user += ("\n---\n" + turn["user_text"]) if merged_user.strip() else turn["user_text"]

            merged_asst = pending["assistant_text"]
            if turn["assistant_text"].strip():
                merged_asst += ("\n---\n" + turn["assistant_text"]) if merged_asst.strip() else turn["assistant_text"]

            merged_tools = pending["tool_summary"]
            if turn["tool_summary"]:
                merged_tools += ("\n" + turn["tool_summary"]) if merged_tools else turn["tool_summary"]

            test_combined = _build_combined(merged_user, merged_asst, merged_tools)
            test_tokens = count_tokens(test_combined)

            if test_tokens <= MAX_CHUNK_TOKENS:
                # Merge succeeds
                pending["user_text"] = merged_user
                pending["assistant_text"] = merged_asst
                pending["tool_summary"] = merged_tools
                pending["tokens"] = test_tokens
                # If merged result reached target size, flush it
                if test_tokens >= MERGE_TARGET_TOKENS:
                    merged.append(pending)
                    pending = None
            else:
                # Would exceed max — flush pending and start fresh
                merged.append(pending)
                if tokens < MERGE_TARGET_TOKENS:
                    pending = {
                        "user_text": turn["user_text"],
                        "assistant_text": turn["assistant_text"],
                        "tool_summary": turn["tool_summary"],
                        "timestamp": turn["timestamp"],
                        "turn_number": i,
                        "tokens": tokens,
                    }
                else:
                    turn["turn_number"] = i
                    merged.append(turn)
                    pending = None

    # Flush any remaining pending turn
    if pending is not None:
        merged.append(pending)

    return merged
