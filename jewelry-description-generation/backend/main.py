import os
import json
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import google.generativeai as genai
from dotenv import load_dotenv
from PIL import Image
import io

# Load environment variables
load_dotenv()

# Configure Gemini
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

app = FastAPI()

# Allow Frontend to talk to Backend (CORS)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # In production, replace with specific frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def analyze_image_with_gemini(image_bytes):
    try:
        # TRY 1: Use the "latest" alias for Flash (often fixes the 404)
        model = genai.GenerativeModel('gemini-2.0-flash')
        
        
        image = Image.open(io.BytesIO(image_bytes))

        prompt = """
        Analyze this image of jewelry. 
        You must return the response in raw JSON format strictly adhering to this structure:
        {
            "type": "Earring" or "Necklace", 
            "description": "A one-line description mentioning the metal type (gold, silver, etc.), gemstone color, shape, and overall style."
        }
        Do not use markdown formatting like ```json. Just return the raw JSON string.
        For any non-jewellery or irrelevant image either choose necklace or earring and write the description based on the colors present and shape and make it seem like you are describing  whatever type you gave. 
        """

        response = model.generate_content([prompt, image])
        
        cleaned_text = response.text.replace('```json', '').replace('```', '').strip()
        
        return json.loads(cleaned_text)
    except Exception as e:
        print(f"Error: {e}")
        # If the model name is still wrong, this will print the available models in your terminal
        print("--- AVAILABLE MODELS ---")
        for m in genai.list_models():
            if 'generateContent' in m.supported_generation_methods:
                print(m.name)
        return None

@app.post("/analyze")
async def analyze_jewelry(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    contents = await file.read()
    result = analyze_image_with_gemini(contents)

    if not result:
        raise HTTPException(status_code=500, detail="Failed to analyze image")

    return result

# To run: uvicorn main:app --reload