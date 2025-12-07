import os
import re
import discord
import anthropic
from dotenv import load_dotenv

load_dotenv()

# Initialize clients
intents = discord.Intents.default()
intents.message_content = True
client = discord.Client(intents=intents)
anthropic_client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

SYSTEM_PROMPT = """You are a participant in a Discord chat. You will receive the last 10 messages from the channel.

You can choose to:
1. Do nothing - output exactly [NO_RESPONSE] and nothing else
2. Respond with a message - just write your response text
3. React to messages with emojis - output [REACT:emoji:message_id] for each reaction

You can combine responses and reactions. For example:
- To just react: [REACT:👍:123456789]
- To react and respond: [REACT:😂:123456789]
That's hilarious!
- Multiple reactions: [REACT:❤️:123456789][REACT:🔥:987654321]

Guidelines:
- Don't respond to every message - only when you have something valuable to add
- Use reactions to acknowledge messages without cluttering the chat
- Be natural and conversational when you do respond
- You are the user "BOT_NAME" in the message history
"""

@client.event
async def on_ready():
    print(f'Logged in as {client.user}')

@client.event
async def on_message(message):
    # Ignore messages from the bot itself
    if message.author == client.user:
        return

    # Fetch last 10 messages
    messages_history = []
    async for msg in message.channel.history(limit=10):
        author_name = "BOT_NAME" if msg.author == client.user else msg.author.display_name
        messages_history.append(f"[{msg.id}] {author_name}: {msg.content}")

    messages_history.reverse()  # Oldest first
    context = "\n".join(messages_history)

    # Call Claude API
    response = anthropic_client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=1024,
        system=SYSTEM_PROMPT.replace("BOT_NAME", client.user.display_name),
        messages=[{"role": "user", "content": f"Recent messages:\n{context}"}]
    )

    response_text = response.content[0].text

    # Check for no response
    if response_text.strip() == "[NO_RESPONSE]":
        return

    # Extract and process reactions
    react_pattern = r'\[REACT:(.+?):(\d+)\]'
    reactions = re.findall(react_pattern, response_text)

    for emoji, msg_id in reactions:
        try:
            target_msg = await message.channel.fetch_message(int(msg_id))
            await target_msg.add_reaction(emoji)
        except Exception as e:
            print(f"Failed to add reaction {emoji} to {msg_id}: {e}")

    # Remove reaction tags from response
    clean_response = re.sub(react_pattern, '', response_text).strip()

    # Send response if there's text remaining
    if clean_response:
        await message.channel.send(clean_response)

# Run the bot
client.run(os.getenv("DISCORD_TOKEN"))
