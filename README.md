# PC Builder Expert Assistant

This application is a conversational AI assistant designed to help users build a custom PC. The assistant guides the user through a series of questions to understand their needs, budget, and preferences. It then uses this information to select a compatible list of PC components, fetches real-time pricing and links from Amazon, and presents the final build to the user in a clear format.

The application uses:
- **Gradio** for the conversational web interface.
- **IBM Watsonx** (via `langchain-ibm`) as the Large Language Model (LLM) to power the conversation.
- **SerpApi** to fetch live product data from Google's Amazon search results.
