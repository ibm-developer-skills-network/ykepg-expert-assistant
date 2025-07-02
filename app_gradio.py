import gradio as gr
from app import get_assistant_response

# Description and title for the Gradio interface
title = "PC Builder Expert Assistant 🤖"
description = """
Welcome! I am your AI-powered PC building expert. 
Tell me what you need and your budget, and I'll recommend a compatible set of parts with prices and links from Amazon.
Start by describing your primary use case (e.g., gaming, video editing, office work) and your budget.
"""

# Create the Gradio Chat Interface
chatbot_ui = gr.ChatInterface(
    fn=get_assistant_response,
    title=title,
    description=description,
    chatbot=gr.Chatbot(
        show_label=False,
        show_copy_button=True,
        render_markdown=True,  # This is crucial for rendering tables and images
    ),
    examples=[
        ["I need a new PC for high-end gaming, my budget is around $2000"],
        ["I want to build a computer for video editing for under $3000"],
        ["I need a cheap computer for my parents, just for Browse and office work."]
    ],
    cache_examples=False,
    retry_btn=None,
    undo_btn="Delete Previous",
    clear_btn="Clear Chat",
)

if __name__ == "__main__":
    # Launch the web application
    chatbot_ui.launch()