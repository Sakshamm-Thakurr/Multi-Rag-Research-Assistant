import google.generativeai as genai
import os
from dotenv import load_dotenv

load_dotenv()

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

print("Available embedding models:\n")
for model in genai.list_models():
    if "embed" in model.name.lower():
        print(f"  Name: {model.name}")
        print(f"  Supported methods: {model.supported_generation_methods}\n")