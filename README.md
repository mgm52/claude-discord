# Claude Discord Bot

An ambient Discord bot powered by Claude that participates in conversations naturally.

## Setup

1. Install dependencies:
   ```
   pip install discord.py anthropic python-dotenv httpx
   ```

2. Create a `.env` file:
   ```
   DISCORD_TOKEN=your_discord_bot_token
   ANTHROPIC_API_KEY=your_anthropic_api_key
   GIPHY_API_KEY=your_giphy_api_key  # Optional, for GIF search
   ```

3. Run the bot:
   ```
   python bot.py
   ```

   Or with auto-restart on file changes:
   ```
   nodemon --exec python bot.py --ext py
   ```
