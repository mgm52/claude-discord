import os
import json
import asyncio
import base64
import httpx
from datetime import datetime
import discord
import anthropic
from dotenv import load_dotenv

load_dotenv()

MEMORY_DIR = "memory"

# Per-channel concurrency control
channel_locks = {}  # channel_id -> asyncio.Lock
channel_pending = {}  # channel_id -> latest pending message (or None)
channel_debounce_tasks = {}  # channel_id -> asyncio.Task for debounce
DEBOUNCE_SECONDS = 3  # Wait this long collecting messages before processing
OUTPUT_NON_MESSAGES = False  # Set to True to show thoughts, web search/fetch, and memory tool usage

# Per-channel check-in mechanism (Claude periodically checks back on the channel)
channel_checkin_tasks = {}  # channel_id -> asyncio.Task for periodic check-in
channel_checkin_delay = {}  # channel_id -> current delay in seconds
CHECKIN_BASE_DELAY = 120  # Initial check-in delay (2 minutes)
CHECKIN_MULTIPLIER = 3  # Multiply delay by this after each check-in

def get_memory_path(guild_id):
    """Get the memory file path for a specific server."""
    os.makedirs(MEMORY_DIR, exist_ok=True)
    return os.path.join(MEMORY_DIR, f"{guild_id}.md")

def read_memory(guild_id):
    """Read the memory file for a server if it exists."""
    try:
        with open(get_memory_path(guild_id), "r", encoding="utf-8") as f:
            return f.read().strip()
    except FileNotFoundError:
        return ""

def write_memory(guild_id, content):
    """Write content to the memory file for a server."""
    with open(get_memory_path(guild_id), "w", encoding="utf-8") as f:
        f.write(content)

# Initialize clients
intents = discord.Intents.default()
intents.message_content = True
# intents.members = True  # Uncomment after enabling "Server Members Intent" in Discord Developer Portal
client = discord.Client(intents=intents)
anthropic_client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

MAX_MESSAGES_CHARS = 2000
MAX_TOOL_ITERATIONS = 10

# Tool definitions for Claude API
TOOLS = [
    # Built-in server-side tools
    {
        "type": "web_search_20250305",
        "name": "web_search",
        "max_uses": 3
    },
    {
        "type": "web_fetch_20250910",
        "name": "web_fetch",
        "max_uses": 3
    },
    # Custom tools
    {
        "name": "send_message",
        "description": "Send a message to the Discord channel. This is the ONLY way to send text that users can see.",
        "input_schema": {
            "type": "object",
            "properties": {
                "content": {
                    "type": "string",
                    "description": "The message content"
                },
                "reply_to_index": {
                    "type": "integer",
                    "description": "Optional: index of a message to reply to (creates a reply with ping)"
                }
            },
            "required": ["content"]
        }
    },
    {
        "name": "add_reaction",
        "description": "Add an emoji reaction to a message.",
        "input_schema": {
            "type": "object",
            "properties": {
                "emoji": {
                    "type": "string",
                    "description": "(e.g., '👍', '❤️', '🔥')"
                },
                "message_index": {
                    "type": "integer",
                    "description": "The index of the message to react to"
                }
            },
            "required": ["emoji", "message_index"]
        }
    },
    {
        "name": "show_typing",
        "description": "Show a typing indicator in the channel. If you intend to do send_message at some point, use this right away.",
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": []
        }
    },
    {
        "name": "memory_append",
        "description": "Append text to your persistent memory file for this server.",
        "input_schema": {
            "type": "object",
            "properties": {
                "text": {
                    "type": "string",
                    "description": "The text to append to memory"
                }
            },
            "required": ["text"]
        }
    },
    {
        "name": "memory_replace",
        "description": "Replace existing text in your memory file.",
        "input_schema": {
            "type": "object",
            "properties": {
                "old_text": {
                    "type": "string",
                    "description": "The exact text to find and replace"
                },
                "new_text": {
                    "type": "string",
                    "description": "The text to replace it with (use empty string to delete)"
                }
            },
            "required": ["old_text", "new_text"]
        }
    },
    {
        "name": "do_nothing",
        "description": "Explicitly choose not to respond or take any action.",
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": []
        }
    },
    {
        "name": "read_image_from_url",
        "description": "Fetch and view an image from a URL. Use this to see images that were shared in the chat (shown in [attachments: ...]). Note: viewing images is costly, so use sparingly.",
        "input_schema": {
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "The URL of the image to view"
                }
            },
            "required": ["url"]
        }
    },
    {
        "name": "view_custom_emoji",
        "description": "View a custom emoji from this server. Use this to see what a custom emoji looks like (custom emojis appear as <:name:id> or <a:name:id> in messages). Note: viewing images is costly, so use sparingly.",
        "input_schema": {
            "type": "object",
            "properties": {
                "emoji_name": {
                    "type": "string",
                    "description": "The name of the custom emoji (without colons)"
                }
            },
            "required": ["emoji_name"]
        }
    },
    {
        "name": "view_sticker",
        "description": "View a sticker that was sent in a message. Stickers appear as [stickers: name] in messages. Note: viewing images is costly, so use sparingly.",
        "input_schema": {
            "type": "object",
            "properties": {
                "sticker_name": {
                    "type": "string",
                    "description": "The name of the sticker to view"
                },
                "message_index": {
                    "type": "integer",
                    "description": "The index of the message containing the sticker"
                }
            },
            "required": ["sticker_name", "message_index"]
        }
    }
]

SYSTEM_PROMPT = """You are a member of a Discord server. You will receive context about the server, channel, and recent messages.

IMPORTANT: Users can ONLY see your tool calls, NOT your text responses. Any text you write outside of tools is invisible to users.
- To send a message: use the send_message tool (optionally with reply_to_index to reply to a specific message)
- To react: use add_reaction
- To stay silent: use do_nothing
- You must use one or more of these user-facing tools (send_message, add_reaction, or do_nothing).

You have a long-term memory file for this server that you can choose to update.
When new messages come in, you will see recent conversation, and your long-term memory.
Try not to exceed 2000 characters of memory in total over time.

Message indices refer to the [N] numbers shown in Recent Messages, where 0 is the most recent message.
However, ONLY you can see the indices; they don't make sense to users, so don't reference indices directly to users.
Existing reactions are shown at the end of messages in brackets, e.g. [👍x3 ❤️] means 3 thumbs up and 1 heart.

Feel free to engage however you'd like. You have no particular pre-defined goals here; you can choose your own goals, if you'd like.
In general, when you're not sure what to do, you could try to mimic the habits / cadence / style of the other members.
Lastly, you don't have to respond every time - just use do_nothing to not engage.
"""

MAX_MEMORY_CHARS = 2000

def build_user_prompt(bot_name, current_time, server_info, channel_info, messages_context, memory):
    if memory:
        memory_section = memory
        if len(memory) > MAX_MEMORY_CHARS:
            memory_section += f"\n\n⚠️ Memory is too long! It's {len(memory)} characters, ideally should be under {MAX_MEMORY_CHARS}. Please clean it up."
    else:
        memory_section = "(empty)"
    return f"""Your nickname: {bot_name}
Current time: {current_time}
{server_info}
{channel_info}

## Your Memory
{memory_section}

## Recent Messages
{messages_context}"""

def save_conversation(system_prompt, messages):
    """Save the full conversation to example_prompt.txt"""
    lines = [f"=== SYSTEM PROMPT ===\n{system_prompt}\n"]

    for msg in messages:
        role = msg["role"].upper()
        content = msg["content"]

        lines.append(f"\n=== {role} ===")
        if isinstance(content, str):
            lines.append(content)
        elif isinstance(content, list):
            for block in content:
                lines.append(format_content_block(block))

    with open("latest_conversation.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def start_checkin_timer(channel):
    """Start a check-in timer for Claude to periodically revisit the channel."""
    channel_id = channel.id

    # Cancel any existing check-in task
    if channel_id in channel_checkin_tasks:
        task = channel_checkin_tasks[channel_id]
        if not task.done():
            task.cancel()

    # Get current delay (or initialize to base)
    if channel_id not in channel_checkin_delay:
        channel_checkin_delay[channel_id] = CHECKIN_BASE_DELAY
    delay = channel_checkin_delay[channel_id]

    async def checkin_process():
        try:
            await asyncio.sleep(delay)
        except asyncio.CancelledError:
            return  # Timer was cancelled (user sent a message or new check-in started)

        # Initialize lock for this channel if needed
        if channel_id not in channel_locks:
            channel_locks[channel_id] = asyncio.Lock()

        lock = channel_locks[channel_id]

        # If channel is busy, skip this check-in
        if lock.locked():
            return

        # Process the channel
        async with lock:
            await process_channel(channel)

        # Increase delay for next check-in (exponential backoff)
        channel_checkin_delay[channel_id] = delay * CHECKIN_MULTIPLIER

    channel_checkin_tasks[channel_id] = asyncio.create_task(checkin_process())


async def execute_single_tool(tool, channel, guild, index_to_id):
    """Execute a single tool and return result string."""
    name = tool.name
    input_data = tool.input
    guild_id = guild.id if guild else "dm"

    try:
        if name == "show_typing":
            await channel.typing()
            return "Typing indicator shown."

        elif name == "send_message":
            content = input_data["content"]
            reply_to_index = input_data.get("reply_to_index")

            if reply_to_index is not None:
                if reply_to_index not in index_to_id:
                    return f"Error: Invalid message index {reply_to_index}. Valid range: 0-{len(index_to_id)-1}"
                real_msg_id = index_to_id[reply_to_index]
                target_msg = await channel.fetch_message(real_msg_id)
                await target_msg.reply(content)
                start_checkin_timer(channel)
                return f"Sent reply to message {reply_to_index}."
            else:
                await channel.send(content)
                start_checkin_timer(channel)
                return "Message sent."

        elif name == "add_reaction":
            emoji = input_data["emoji"]
            msg_index = input_data["message_index"]

            if msg_index not in index_to_id:
                return f"Error: Invalid message index {msg_index}. Valid range: 0-{len(index_to_id)-1}"

            real_msg_id = index_to_id[msg_index]
            target_msg = await channel.fetch_message(real_msg_id)
            await target_msg.add_reaction(emoji)
            return f"Added reaction {emoji} to message {msg_index}."

        elif name == "memory_append":
            text = input_data["text"]
            current_memory = read_memory(guild_id)
            new_memory = current_memory + "\n" + text if current_memory else text
            write_memory(guild_id, new_memory)
            if OUTPUT_NON_MESSAGES:
                await channel.send(f'*used memory_append: "{text}"*')
            return f"Appended to memory."

        elif name == "memory_replace":
            old_text = input_data["old_text"]
            new_text = input_data["new_text"]
            current_memory = read_memory(guild_id)

            if old_text not in current_memory:
                return f"Error: Text not found in memory: '{old_text[:50]}...'"

            updated_memory = current_memory.replace(old_text, new_text)
            write_memory(guild_id, updated_memory)
            if OUTPUT_NON_MESSAGES:
                if new_text:
                    await channel.send(f'*used memory_replace: "{old_text}" → "{new_text}"*')
                else:
                    await channel.send(f'*used memory_replace to delete: "{old_text}"*')
            return f"Replaced in memory."

        elif name == "do_nothing":
            return "No action taken."

        elif name == "read_image_from_url":
            url = input_data["url"]
            async with httpx.AsyncClient() as http_client:
                response = await http_client.get(url, follow_redirects=True)
                response.raise_for_status()
                content_type = response.headers.get("content-type", "image/png")
                # Extract just the mime type (e.g., "image/png" from "image/png; charset=utf-8")
                if ";" in content_type:
                    content_type = content_type.split(";")[0].strip()
                image_data = base64.b64encode(response.content).decode("utf-8")
                return [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": content_type,
                            "data": image_data
                        }
                    },
                    {"type": "text", "text": f"Image from {url}"}
                ]

        elif name == "view_custom_emoji":
            emoji_name = input_data["emoji_name"]
            if not guild:
                return "Error: Cannot view custom emojis in DMs"

            # Find the emoji by name in the guild
            emoji = discord.utils.get(guild.emojis, name=emoji_name)
            if not emoji:
                available = [e.name for e in guild.emojis[:10]]
                return f"Error: Emoji '{emoji_name}' not found. Available emojis include: {', '.join(available)}"

            # Fetch the emoji image
            emoji_url = str(emoji.url)
            async with httpx.AsyncClient() as http_client:
                response = await http_client.get(emoji_url, follow_redirects=True)
                response.raise_for_status()
                content_type = response.headers.get("content-type", "image/png")
                if ";" in content_type:
                    content_type = content_type.split(";")[0].strip()
                image_data = base64.b64encode(response.content).decode("utf-8")
                return [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": content_type,
                            "data": image_data
                        }
                    },
                    {"type": "text", "text": f"Custom emoji :{emoji_name}: (animated: {emoji.animated})"}
                ]

        elif name == "view_sticker":
            sticker_name = input_data["sticker_name"]
            msg_index = input_data["message_index"]

            if msg_index not in index_to_id:
                return f"Error: Invalid message index {msg_index}. Valid range: 0-{len(index_to_id)-1}"

            # Fetch the message to get stickers
            real_msg_id = index_to_id[msg_index]
            target_msg = await channel.fetch_message(real_msg_id)

            if not target_msg.stickers:
                return f"Error: Message {msg_index} has no stickers"

            # Find the sticker by name
            sticker = None
            for s in target_msg.stickers:
                if s.name.lower() == sticker_name.lower():
                    sticker = s
                    break

            if not sticker:
                available = [s.name for s in target_msg.stickers]
                return f"Error: Sticker '{sticker_name}' not found in message. Available: {', '.join(available)}"

            # Check if it's a Lottie sticker (can't view as image)
            if "lottie" in str(sticker.format).lower():
                return f"Sticker '{sticker_name}' is a Lottie animation (JSON format) and cannot be viewed as an image. It's an animated sticker."

            # Fetch the sticker image
            sticker_url = str(sticker.url)
            async with httpx.AsyncClient() as http_client:
                response = await http_client.get(sticker_url, follow_redirects=True)
                response.raise_for_status()
                content_type = response.headers.get("content-type", "image/png")
                if ";" in content_type:
                    content_type = content_type.split(";")[0].strip()
                image_data = base64.b64encode(response.content).decode("utf-8")
                return [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": content_type,
                            "data": image_data
                        }
                    },
                    {"type": "text", "text": f"Sticker: {sticker_name}"}
                ]

        else:
            return f"Error: Unknown tool '{name}'"

    except Exception as e:
        print(f"Error executing {name}: {e}")
        return f"Error executing {name}: {str(e)}"


async def execute_tools(tool_uses, channel, guild, index_to_id):
    """
    Execute tool calls with proper ordering.
    Order: typing -> memory -> reactions -> messages -> do_nothing
    """
    # Categorize tools by priority
    typing_tools = []
    memory_tools = []
    reaction_tools = []
    message_tools = []
    noop_tools = []

    for tool in tool_uses:
        name = tool.name
        if name == "show_typing":
            typing_tools.append(tool)
        elif name in ("memory_append", "memory_replace"):
            memory_tools.append(tool)
        elif name == "add_reaction":
            reaction_tools.append(tool)
        elif name == "send_message":
            message_tools.append(tool)
        elif name == "do_nothing":
            noop_tools.append(tool)
        else:
            # Unknown tool, still try to execute
            message_tools.append(tool)

    # Execute in order and collect results
    results = []
    ordered_tools = typing_tools + memory_tools + reaction_tools + message_tools + noop_tools

    for tool in ordered_tools:
        result = await execute_single_tool(tool, channel, guild, index_to_id)
        results.append({
            "type": "tool_result",
            "tool_use_id": tool.id,
            "content": result
        })

    return results


def format_content_block(block):
    """Format a content block for logging."""
    if hasattr(block, 'type'):
        if block.type == "text":
            return f"[text] {block.text}"
        elif block.type == "tool_use":
            return f"[tool_use] {block.name}({json.dumps(block.input)})"
    elif isinstance(block, dict):
        if block.get("type") == "tool_result":
            return f"[tool_result] {block['content']}"
    return str(block)

class OutOfCreditsError(Exception):
    """Raised when Anthropic API returns a credit/billing error."""
    pass


async def handle_claude_response(user_prompt, channel, guild, index_to_id):
    """
    Handle Claude API response with tool use loop.
    Calls Claude, executes any tools, and loops until done.
    Returns the full message history for logging.
    """
    messages = [{"role": "user", "content": user_prompt}]
    user_facing_tools = {"send_message", "add_reaction", "do_nothing"}
    server_side_tools = {"web_search", "web_fetch"}
    used_user_facing_tool = False

    def call_claude_api(msgs):
        """Call Claude API and handle credit errors."""
        try:
            return anthropic_client.messages.create(
                model="claude-haiku-4-5-20251001",
                max_tokens=1024,
                system=SYSTEM_PROMPT,
                tools=TOOLS,
                messages=msgs,
                extra_headers={"anthropic-beta": "web-fetch-2025-09-10"}
            )
        except anthropic.APIStatusError as e:
            # Check for credit/billing/usage limit related errors
            error_msg = str(e).lower()
            if any(phrase in error_msg for phrase in ["credit", "billing", "payment", "usage limit"]) or e.status_code == 402:
                raise OutOfCreditsError(str(e))
            raise

    for iteration in range(MAX_TOOL_ITERATIONS):
        response = call_claude_api(messages)

        # Collect tool uses from response (both custom and server-side)
        tool_uses = [block for block in response.content if block.type == "tool_use"]
        server_tool_uses = [block for block in response.content if block.type == "server_tool_use"]

        # Track if any user-facing tools were used
        for tool in tool_uses:
            if tool.name in user_facing_tools:
                used_user_facing_tool = True

        # Show server-side tool notifications (if enabled)
        if OUTPUT_NON_MESSAGES:
            for tool in server_tool_uses:
                if tool.name == "web_search":
                    query = tool.input.get("query", "")
                    await channel.send(f'*searched the web for: "{query}"*')
                elif tool.name == "web_fetch":
                    url = tool.input.get("url", "")
                    await channel.send(f'*fetched: {url}*')

        # Filter to only custom tools that we need to execute (server-side tools are handled automatically by the API)
        custom_tool_uses = [t for t in tool_uses if t.name not in server_side_tools]

        # If there are custom tools to execute
        if custom_tool_uses:
            tool_results = await execute_tools(custom_tool_uses, channel, guild, index_to_id)

            # Add assistant response and tool results to messages for next iteration
            messages.append({"role": "assistant", "content": response.content})
            messages.append({"role": "user", "content": tool_results})
        elif server_tool_uses or any(block.type in ("web_search_tool_result", "web_fetch_tool_result") for block in response.content):
            # Server-side tools were used or results returned - add response and continue loop
            messages.append({"role": "assistant", "content": response.content})

        # Check if we're done
        if response.stop_reason == "end_turn":
            messages.append({"role": "assistant", "content": response.content})
            break
        elif response.stop_reason != "tool_use":
            print(f"Unexpected stop_reason: {response.stop_reason}")
            messages.append({"role": "assistant", "content": response.content})
            break

    # If Claude didn't use any user-facing tool, send a reminder and let it try again
    if not used_user_facing_tool:
        reminder = (
            "REMINDER: Your turn ended without using any user-facing tool. "
            "Users cannot see your text responses - only tool calls are visible. "
            "You MUST use one of: send_message (to send a message), add_reaction (to react), or do_nothing (to stay silent). "
            "Please choose one now."
        )
        messages.append({"role": "user", "content": reminder})

        # Give Claude one more chance
        response = call_claude_api(messages)

        tool_uses = [block for block in response.content if block.type == "tool_use"]
        custom_tool_uses = [t for t in tool_uses if t.name not in server_side_tools]

        if custom_tool_uses:
            tool_results = await execute_tools(custom_tool_uses, channel, guild, index_to_id)
            messages.append({"role": "assistant", "content": response.content})
            messages.append({"role": "user", "content": tool_results})
        else:
            messages.append({"role": "assistant", "content": response.content})

    return messages


@client.event
async def on_ready():
    print(f'Logged in as {client.user}')


async def process_channel(channel):
    """Process the channel - gathers fresh context and calls Claude."""
    # Gather server info
    guild = channel.guild
    if guild:
        server_info = f"Server name: {guild.name}\nMembers: {guild.member_count}"
        if guild.description:
            server_info += f"\nDescription: {guild.description}"
    else:
        server_info = "Direct Message (no server)"

    # Gather channel info
    channel_info = f"Current channel: #{channel.name}" if hasattr(channel, 'name') else "(Direct Message)"
    if hasattr(channel, 'topic') and channel.topic:
        channel_info += f"\nTopic: {channel.topic}"
    if hasattr(channel, 'category') and channel.category:
        channel_info += f"\nCategory: {channel.category.name}"

    # Fetch messages and build context within character limit
    bot_name = client.user.display_name
    all_messages = []
    messages_by_id = {}  # Track messages by ID for reply lookups

    async for msg in channel.history(limit=50):
        author_name = bot_name if msg.author == client.user else msg.author.display_name
        timestamp = msg.created_at.strftime("%Y-%m-%d %H:%M")
        # Collect reactions as "emoji xN" format
        reactions = []
        for reaction in msg.reactions:
            emoji_str = str(reaction.emoji)
            if reaction.count > 1:
                reactions.append(f"{emoji_str}x{reaction.count}")
            else:
                reactions.append(emoji_str)
        msg_data = {
            'id': msg.id,
            'author': author_name,
            'timestamp': timestamp,
            'content': msg.content,
            'reply_to_id': msg.reference.message_id if msg.reference else None,
            'reactions': reactions,
            'attachments': [{'filename': a.filename, 'url': a.url, 'content_type': a.content_type}
                           for a in msg.attachments],
            'stickers': [{'name': s.name, 'id': s.id, 'url': str(s.url), 'format': str(s.format)}
                        for s in msg.stickers]
        }
        all_messages.append(msg_data)
        messages_by_id[msg.id] = msg_data

    # all_messages is newest first from API - keep it that way
    # Build message context within character limit, index 0 = most recent
    messages_to_include = []
    total_chars = 0
    index_to_id = {}  # Map index to real message ID
    id_to_index = {}  # Map real message ID to index

    for i, msg in enumerate(all_messages):
        reply_part = ""
        if msg['reply_to_id']:
            reply_part = f" [reply to msg {msg['reply_to_id']}]"
        line = f"[{i}] {msg['author']} ({msg['timestamp']}){reply_part}: {msg['content']}\n"
        if total_chars + len(line) > MAX_MESSAGES_CHARS:
            break
        messages_to_include.append(msg)
        index_to_id[i] = msg['id']
        id_to_index[msg['id']] = i
        total_chars += len(line)

    # Find referenced messages that aren't in context and need to be fetched
    referenced_ids_to_fetch = set()
    for msg in messages_to_include:
        if msg['reply_to_id'] and msg['reply_to_id'] not in id_to_index:
            referenced_ids_to_fetch.add(msg['reply_to_id'])

    # Fetch and add missing referenced messages
    referenced_messages = []
    for ref_id in referenced_ids_to_fetch:
        # Check if we already have it from history
        if ref_id in messages_by_id:
            referenced_messages.append(messages_by_id[ref_id])
        else:
            # Need to fetch it
            try:
                ref_msg = await channel.fetch_message(ref_id)
                author_name = bot_name if ref_msg.author == client.user else ref_msg.author.display_name
                timestamp = ref_msg.created_at.strftime("%Y-%m-%d %H:%M")
                reactions = []
                for reaction in ref_msg.reactions:
                    emoji_str = str(reaction.emoji)
                    if reaction.count > 1:
                        reactions.append(f"{emoji_str}x{reaction.count}")
                    else:
                        reactions.append(emoji_str)
                referenced_messages.append({
                    'id': ref_msg.id,
                    'author': author_name,
                    'timestamp': timestamp,
                    'content': ref_msg.content,
                    'reply_to_id': ref_msg.reference.message_id if ref_msg.reference else None,
                    'reactions': reactions,
                    'attachments': [{'filename': a.filename, 'url': a.url, 'content_type': a.content_type}
                                   for a in ref_msg.attachments],
                    'stickers': [{'name': s.name, 'id': s.id, 'url': str(s.url), 'format': str(s.format)}
                                for s in ref_msg.stickers]
                })
            except Exception as e:
                print(f"Could not fetch referenced message {ref_id}: {e}")

    # Assign indices to referenced messages (continuing from where we left off)
    next_index = len(messages_to_include)
    for ref_msg in referenced_messages:
        index_to_id[next_index] = ref_msg['id']
        id_to_index[ref_msg['id']] = next_index
        next_index += 1

    # Build context lines (reversed so oldest appears first in display, but indices stay same)
    context_lines = []
    for i in range(len(messages_to_include) - 1, -1, -1):
        msg = messages_to_include[i]
        reply_part = ""
        if msg['reply_to_id']:
            # Now we can reference by index if available
            if msg['reply_to_id'] in id_to_index:
                reply_part = f" [replying to {id_to_index[msg['reply_to_id']]}]"
            else:
                reply_part = f" [replying to deleted/unknown msg]"
        reactions_part = ""
        if msg.get('reactions'):
            reactions_part = f" [{' '.join(msg['reactions'])}]"
        attachments_part = ""
        if msg.get('attachments'):
            att_strs = [f"{a['filename']}: {a['url']}" for a in msg['attachments']]
            attachments_part = " [attachments: " + ", ".join(att_strs) + "]"
        stickers_part = ""
        if msg.get('stickers'):
            sticker_strs = [s['name'] for s in msg['stickers']]
            stickers_part = " [stickers: " + ", ".join(sticker_strs) + "]"
        context_lines.append(f"[{i}] {msg['author']} ({msg['timestamp']}){reply_part}: {msg['content']}{reactions_part}{attachments_part}{stickers_part}")

    # Add referenced messages section if any
    if referenced_messages:
        context_lines.append("\n--- Referenced messages (older, pulled into context) ---")
        for ref_msg in referenced_messages:
            idx = id_to_index[ref_msg['id']]
            reply_part = ""
            if ref_msg['reply_to_id'] and ref_msg['reply_to_id'] in id_to_index:
                reply_part = f" [replying to {id_to_index[ref_msg['reply_to_id']]}]"
            reactions_part = ""
            if ref_msg.get('reactions'):
                reactions_part = f" [{' '.join(ref_msg['reactions'])}]"
            attachments_part = ""
            if ref_msg.get('attachments'):
                att_strs = [f"{a['filename']}: {a['url']}" for a in ref_msg['attachments']]
                attachments_part = " [attachments: " + ", ".join(att_strs) + "]"
            stickers_part = ""
            if ref_msg.get('stickers'):
                sticker_strs = [s['name'] for s in ref_msg['stickers']]
                stickers_part = " [stickers: " + ", ".join(sticker_strs) + "]"
            context_lines.append(f"[{idx}] {ref_msg['author']} ({ref_msg['timestamp']}){reply_part}: {ref_msg['content']}{reactions_part}{attachments_part}{stickers_part}")

    context = "\n".join(context_lines)

    # Build prompt and call Claude API
    current_time = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M %Z")
    guild_id = guild.id if guild else "dm"
    memory = read_memory(guild_id)
    user_prompt = build_user_prompt(bot_name, current_time, server_info, channel_info, context, memory)

    try:
        messages = await handle_claude_response(user_prompt, channel, guild, index_to_id)
        save_conversation(SYSTEM_PROMPT, messages)
    except OutOfCreditsError as e:
        print(f"Out of API credits: {e}")
        # Only send apology if the latest message mentions the bot directly
        async for msg in channel.history(limit=1):
            if client.user in msg.mentions:
                await channel.send("sorry, i'm seeing an error saying there are no anthropic API credits left, so i can't respond properly")
            break
    except Exception as e:
        import traceback
        timestamp = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")
        error_msg = f"[{timestamp}] Error handling Claude response: {e}\n{traceback.format_exc()}"
        print(error_msg)
        # Write full error to file for debugging
        with open("error_log.txt", "w", encoding="utf-8") as f:
            f.write(error_msg)


@client.event
async def on_message(message):
    # Ignore messages from the bot itself
    if message.author == client.user:
        return

    channel_id = message.channel.id
    channel = message.channel

    # Cancel any existing debounce task for this channel (newer message replaces it)
    if channel_id in channel_debounce_tasks:
        task = channel_debounce_tasks[channel_id]
        if not task.done():
            task.cancel()

    # Cancel any existing check-in task and reset the backoff (user interaction resets everything)
    if channel_id in channel_checkin_tasks:
        task = channel_checkin_tasks[channel_id]
        if not task.done():
            task.cancel()
    channel_checkin_delay[channel_id] = CHECKIN_BASE_DELAY  # Reset to base delay

    # Create a debounced processing task
    async def debounced_process():
        try:
            await asyncio.sleep(DEBOUNCE_SECONDS)  # Wait, collecting newer messages
        except asyncio.CancelledError:
            return  # Newer message came in, this task was replaced

        # Initialize lock for this channel if needed
        if channel_id not in channel_locks:
            channel_locks[channel_id] = asyncio.Lock()

        lock = channel_locks[channel_id]

        # Try to acquire the lock without blocking
        if lock.locked():
            # Channel is busy - store this as the pending trigger (replaces any previous)
            channel_pending[channel_id] = channel
            return

        # We can process - acquire lock and go
        async with lock:
            await process_channel(channel)

            # After processing, check if there's a pending message for this channel
            # Keep processing until no more pending
            while channel_id in channel_pending:
                pending_channel = channel_pending.pop(channel_id)
                await process_channel(pending_channel)

    channel_debounce_tasks[channel_id] = asyncio.create_task(debounced_process())


# Run the bot
client.run(os.getenv("DISCORD_TOKEN"))
