from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os
import logging
import json
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI

# Load OpenAI API key
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise RuntimeError("Missing OPENAI_API_KEY environment variable")

# Initialize OpenAI client
client = OpenAI(api_key=api_key)

# Create FastAPI app
app = FastAPI(title="Text Analysis API")

# Enable CORS for all origins (for testing; restrict to specific domains in production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic schemas
class AnalysisRequest(BaseModel):
    text: str

class AnalysisResponse(BaseModel):
    emotion: int
    factuality: int
    notes: str

# Endpoint for text analysis
@app.post("/analyze", response_model=AnalysisResponse)
async def analyze(req: AnalysisRequest):
    try:
        # Strict prompt requesting clean JSON output only
        prompt = f"""

Analyze the following text using these three criteria:\\

\noindent 1. 
1. **Emotionality**: Assess how emotionally charged or expressive the language is. Use a scale from -5 (neutral, analytical) to +5 (strongly emotional, biased, sarcastic).

2. **Factuality vs. Speculativeness**: Evaluate how much the text relies on facts, scientific sources, logical reasoning, or verified data. If it lacks supporting evidence, relies on speculation or conspiracy framing, score closer to +5. If it is well-referenced and evidence-based, score closer to -5.

3. **Notes**: Identify any argumentative fallacies (e.g., ad hominem, strawman, false cause, slippery slope, etc.). If no significant fallacies are found, reply with "No obvious argumentative fallacies"

Example: The claim "scientists are corrupt and follow an agenda" contains both ad hominem and conspiracy framing.

Text to analyze:
'''{req.text}'''

---

Respond with a valid JSON object in this format:

{{
  "emotion": <number between –5 and +5>,
  "factuality": <number between –5 and +5>,
  "notes": "<list of fallacies or 'No apparent logical fallacies'>"
}}

Do not write any additional text, comments, intros, or explanations. Return only a valid JSON response.
"""

        # Call the new OpenAI v1 API
        response = client.chat.completions.create(
            model="gpt-4-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=256
        )

        # Extract response content
        content = response.choices[0].message.content
        logging.info("LLM raw response: %s", content)

        # Attempt to parse the JSON
        try:
            data = json.loads(content)
        except json.JSONDecodeError:
            logging.error("Invalid JSON from LLM: %s", content)
            raise HTTPException(
                status_code=502,
                detail=f"Invalid response from LLM: {content}"
            )

        # Return structured response
        return AnalysisResponse(
            emotion=int(data.get("emotion", 0)),
            factuality=int(data.get("factuality", 0)),
            notes=data.get("notes", "")
        )

    except HTTPException:
        # Rethrow HTTP exceptions
        raise
    except Exception as e:
        logging.exception("Error processing the request:")
        raise HTTPException(status_code=500, detail=str(e))
