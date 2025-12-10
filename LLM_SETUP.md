# Diabetes Risk Checker - Setup Instructions

## LLM-Powered Health Suggestions

The app now uses OpenAI's GPT-4o-mini to generate personalized, intelligent health suggestions.

### Setup:

1. **Install the OpenAI package:**
   ```powershell
   pip install openai
   ```

2. **Get your OpenAI API key:**
   - Go to https://platform.openai.com/api-keys
   - Create a new API key
   - Copy the key

3. **Add your API key:**
   - Open `.streamlit/secrets.toml`
   - Replace `"your-api-key-here"` with your actual API key:
   ```toml
   OPENAI_API_KEY = "sk-proj-..."
   ```

4. **Run the app:**
   ```powershell
   streamlit run diabetes_app.py
   ```

### Features:

- **AI-Generated Suggestions**: Personalized advice based on your specific health data
- **Fallback Mode**: If no API key is configured, uses rule-based suggestions
- **Cost-Effective**: Uses gpt-4o-mini model (~$0.15 per 1M tokens)

### Without API Key:

The app will still work with basic rule-based suggestions. Add the API key anytime to unlock AI-powered recommendations.
