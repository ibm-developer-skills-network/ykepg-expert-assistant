import os
import re
from dotenv import load_dotenv
from langchain_ibm import WatsonxLLM
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field
from serpapi import GoogleSearch
import json

# --- ENVIRONMENT AND API SETUP ---
load_dotenv()

# Check for API Keys
if not os.getenv("WATSONX_API_KEY") or not os.getenv("SERPAPI_API_KEY") or not os.getenv("WATSONX_PROJECT_ID"):
    raise ValueError("API keys for Watsonx or SerpApi are not set in the .env file.")

# Llama Model Configuration
LLM = WatsonxLLM(
    model_id="meta-llama/llama-3-1-70b-instruct",
    url="https://us-south.ml.cloud.ibm.com",
    project_id=os.getenv("WATSONX_PROJECT_ID"),
    params={"decoding_method": "greedy", "max_new_tokens": 1024, "repetition_penalty": 1}
)

# --- EXPERT SYSTEM: PC PART DATA ---
# This dictionary contains pre-defined, compatible PC builds based on use case and budget.
# This rule-based approach ensures component compatibility.
PC_BUILDS = {
    "budget_office": {
        "name": "Budget Office/Web Browse Build",
        "parts": {
            "CPU": "AMD Ryzen 5 5600G",
            "Motherboard": "ASRock B450M PRO4",
            "RAM": "Corsair Vengeance LPX 16GB DDR4 3200MHz",
            "Storage": "Crucial P3 1TB NVMe SSD",
            "PSU": "Thermaltake Smart 500W 80+ White",
            "Case": "Cooler Master MasterBox Q300L",
        }
    },
    "mid_range_gaming": {
        "name": "Mid-Range Gaming Build",
        "parts": {
            "CPU": "AMD Ryzen 5 7600X",
            "GPU": "NVIDIA GeForce RTX 4060",
            "Motherboard": "Gigabyte B650 Gaming X AX",
            "RAM": "G.Skill Flare X5 32GB DDR5 6000MHz",
            "Storage": "Samsung 980 Pro 1TB NVMe SSD",
            "CPU Cooler": "Thermalright Phantom Spirit 120 SE",
            "PSU": "Corsair RM750e 750W 80+ Gold",
            "Case": "Lian Li Lancool 216",
        }
    },
    "high_end_gaming": {
        "name": "High-End Gaming/Streaming Build",
        "parts": {
            "CPU": "AMD Ryzen 7 7800X3D",
            "GPU": "NVIDIA GeForce RTX 4070 Super",
            "Motherboard": "MSI MAG B650 Tomahawk WiFi",
            "RAM": "G.Skill Trident Z5 Neo 32GB DDR5 6000MHz CL30",
            "Storage": "Western Digital Black SN850X 2TB NVMe SSD",
            "CPU Cooler": "Thermalright Phantom Spirit 120 SE",
            "PSU": "SeaSonic FOCUS Plus Gold 850W",
            "Case": "Fractal Design Pop Air",
        }
    },
    "professional_workstation": {
        "name": "Professional Video Editing Workstation",
        "parts": {
            "CPU": "Intel Core i7-14700K",
            "GPU": "NVIDIA GeForce RTX 4080 Super",
            "Motherboard": "MSI PRO Z790-A WIFI",
            "RAM": "Corsair Vengeance 64GB DDR5 5600MHz",
            "Storage": "Samsung 990 Pro 2TB NVMe SSD",
            "CPU Cooler": "ARCTIC Liquid Freezer II 360",
            "PSU": "Corsair RM1000e 1000W 80+ Gold",
            "Case": "be quiet! Pure Base 500DX",
        }
    }
}

# --- LLM PROMPTING AND PARSING ---

class NeedsAnalysis(BaseModel):
    budget: int = Field(description="The user's approximate budget as an integer.")
    use_case: str = Field(description="The primary use case, categorized as 'gaming', 'office', 'editing', or 'unknown'.")
    has_confirmed: bool = Field(description="True if the user has explicitly confirmed the details (e.g., said 'yes', 'correct').")

json_parser = JsonOutputParser(pydantic_object=NeedsAnalysis)

llama_template = PromptTemplate(
    template="System: {system_prompt}\n{format_prompt}\nHuman: Here is the chat history:\n{chat_history}\nAI:",
    input_variables=["system_prompt", "format_prompt", "chat_history"]
)

def analyze_user_needs(chat_history_str: str) -> dict:
    """Uses the LLM to analyze the conversation and extract key user requirements."""
    system_prompt = (
        "You are an expert at analyzing chatbot conversations. Your task is to extract the user's budget and primary PC use case. "
        "Categorize the use case into one of these specific keywords: 'gaming', 'office', 'editing', or 'unknown'. "
        "Determine the budget as an integer. Also, determine if the user has just confirmed the plan by saying 'yes' or 'correct'."
    )
    format_prompt = json_parser.get_format_instructions()
    chain = llama_template | LLM | json_parser
    
    try:
        response = chain.invoke({
            "system_prompt": system_prompt,
            "format_prompt": format_prompt,
            "user_prompt": chat_history_str
        })
        return response
    except Exception as e:
        print(f"Error analyzing user needs: {e}")
        return {"budget": 0, "use_case": "unknown", "has_confirmed": False}

# --- EXTERNAL API FUNCTIONS ---

def search_amazon_for_component(part_name: str) -> dict:
    """Searches Amazon for a specific PC part and returns its details."""
    print(f"Searching for: {part_name}")
    params = {
        "engine": "google",
        "q": f"amazon.com {part_name}",
        "tbm": "shop",
        "num": 1,
        "api_key": os.getenv("SERPAPI_API_KEY")
    }
    search = GoogleSearch(params)
    results = search.get_dict()
    shopping_results = results.get("shopping_results", [])

    if shopping_results:
        product = shopping_results[0]
        return {
            "name": product.get("title", part_name),
            "price": product.get("price", "N/A"),
            "link": product.get("link", "#"),
            "image": product.get("thumbnail", "")
        }
    return {"name": part_name, "price": "Not Found", "link": "#", "image": ""}

# --- CORE LOGIC ---

def select_build_tier(budget: int, use_case: str) -> dict:
    """Selects the appropriate build tier based on user input."""
    if "gaming" in use_case:
        if budget < 1600:
            return PC_BUILDS["mid_range_gaming"]
        return PC_BUILDS["high_end_gaming"]
    elif "editing" in use_case:
        return PC_BUILDS["professional_workstation"]
    elif "office" in use_case:
        return PC_BUILDS["budget_office"]
    # Fallback based on budget if use case is unclear
    if budget <= 700:
        return PC_BUILDS["budget_office"]
    elif budget <= 1600:
        return PC_BUILDS["mid_range_gaming"]
    else:
        return PC_BUILDS["high_end_gaming"]

def format_results_as_markdown(build: dict, part_details: list) -> str:
    """Formats the final list of components into a Markdown table."""
    total_price = 0
    markdown_table = f"### Your Recommended {build['name']}\n\n"
    markdown_table += "| Component | Part | Price | Image |\n"
    markdown_table += "|---|---|---|---|\n"

    for part in part_details:
        price_str = re.sub(r'[^\d.]', '', part['price'])
        if price_str:
            total_price += float(price_str)
        
        markdown_table += (
            f"| **{part['type']}** | [{part['name']}]({part['link']}) "
            f"| {part['price']} "
            f"| ![{part['name']}]({part['image']}) |\n"
        )
    
    markdown_table += f"\n**Estimated Total: ${total_price:.2f}**"
    markdown_table += "\n\n*Prices are estimates from live search results and may vary. Links will open in a new tab.*"
    return markdown_table

def get_assistant_response(user_message: str, chat_history: list) -> str:
    """
    Main function to process user input and generate the assistant's response.
    This is the primary entry point called by the Gradio UI.
    """
    if not chat_history:
        return "Hello! I'm your PC building expert assistant. To get started, what do you plan to use this PC for and what is your approximate budget?"

    # Convert chat history to a string for the LLM
    history_str = "\n".join([f"User: {turn[0]}\nAssistant: {turn[1]}" for turn in chat_history])
    history_str += f"\nUser: {user_message}"
    
    # Analyze the conversation to understand current needs
    needs = analyze_user_needs(history_str)
    
    # 1. Information Gathering
    if needs['budget'] == 0:
        return "I see. And what is your approximate budget for the new PC?"
    if needs['use_case'] == 'unknown':
        return f"Got it, a budget of around ${needs['budget']}. What is the primary use for this PC? For example: high-end gaming, video editing, or just office work and web Browse?"

    # 2. Confirmation
    if not needs['has_confirmed']:
        return f"Okay, just to confirm: you're looking for a PC for **{needs['use_case']}** with a budget around **${needs['budget']}**. Is that correct?"

    # 3. Component Selection and Final Output
    print("User confirmed. Starting build process...")
    selected_build = select_build_tier(needs['budget'], needs['use_case'])
    
    final_parts_list = []
    for part_type, part_name in selected_build["parts"].items():
        search_result = search_amazon_for_component(part_name)
        search_result['type'] = part_type
        final_parts_list.append(search_result)
        
    if not final_parts_list:
        return "I'm sorry, I encountered an error while searching for parts. Please try again."
        
    return format_results_as_markdown(selected_build, final_parts_list)