from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# Import the routers from separate modules based on your directory structure
from bank.qa_routes import qa_router
from translation.translate_routes import translation_router
from tts.tts_routes import tts_router
from category.classifer_routes import router as classifier_router

# Create main FastAPI app
app = FastAPI(title="Multi-Service AI API", version="1.0.0")

# Enable CORS for all domains
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # You can restrict origins here
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include all routers
app.include_router(qa_router)
app.include_router(translation_router)
app.include_router(tts_router)
app.include_router(classifier_router)

@app.get("/")
async def root():
    return {
        "message": "Multi-Service AI API",
        "services": {
            "qa": "/predict - Question Answering",
            "translation": "/translate - English to Telugu Translation", 
            "tts": "/tts - Text to Speech"
        }
    }

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=5001, reload=True)