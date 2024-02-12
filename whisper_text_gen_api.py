from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
from whisper_text_gen_service import TextGeneratorService

app = FastAPI()

text_generator_service = TextGeneratorService()

class GenerateTextRequest(BaseModel):
    seed_text: str
    num_generate: int

@app.post("/generate-text/")
async def generate_text(request: GenerateTextRequest):
    try:
        generated_text = text_generator_service.generate_text(request.seed_text, request.num_generate)
        return {"generated_text": generated_text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5050)
