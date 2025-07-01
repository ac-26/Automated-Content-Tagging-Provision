from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from typing import List, Optional, Tuple
import uvicorn
from datetime import datetime
import time
import sys

print("Starting app.py imports...", flush=True)

# Import your tagger classes
try:
    print("Importing DynamicTagger modules...", flush=True)
    from dynamic_tagger import DynamicTagger, TextEncoder, KeyPhraseExtractor, PhraseScorer
    print("Successfully imported tagger modules!", flush=True)
except Exception as e:
    print(f"Error importing tagger modules: {e}", flush=True)
    raise

# Initialize FastAPI app
print("Initializing FastAPI app...", flush=True)
app = FastAPI(
    title="Dynamic Content Tagger API",
    description="AI-powered content tagging system that works with any domain",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Initialize the tagger
print("Initializing Dynamic Tagger...", flush=True)
print("This may take a few minutes as models are downloaded...", flush=True)
try:
    tagger = DynamicTagger()
    print("Tagger initialized successfully!", flush=True)
except Exception as e:
    print(f"Error initializing tagger: {e}", flush=True)
    raise

# Pydantic models (keep your existing ones)
class TagRequest(BaseModel):
    text: str = Field(..., description="Text content to generate tags from", min_length=10)
    max_tags: Optional[int] = Field(7, description="Maximum number of tags to return", ge=1, le=20)
    min_score: Optional[float] = Field(0.6, description="Minimum score threshold for tags", ge=0.0, le=1.0)
    include_scores: Optional[bool] = Field(False, description="Whether to include confidence scores with tags")

class TagResponse(BaseModel):
    tags: List[str]
    scores: Optional[List[float]] = None
    processing_time: float
    tag_count: int
    timestamp: datetime

class TagWithScoreResponse(BaseModel):
    tags: List[dict]
    processing_time: float
    tag_count: int
    timestamp: datetime

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    timestamp: datetime

# Root endpoint - serve the frontend
@app.get("/")
async def read_root():
    return FileResponse('static/index.html')

# Explicitly serve CSS
@app.get("/style.css")
async def get_css():
    return FileResponse('static/style.css', media_type='text/css')

# Explicitly serve JS
@app.get("/script.js") 
async def get_js():
    return FileResponse('static/script.js', media_type='application/javascript')


@app.get("/api", response_model=dict)
async def api_root():
    """API information endpoint"""
    return {
        "message": "Welcome to Dynamic Content Tagger API",
        "version": "2.0.0",
        "endpoints": {
            "/docs": "Interactive API documentation",
            "/health": "Check API health status",
            "/tag": "Generate tags from text (POST)",
            "/tag/detailed": "Generate tags with detailed scores (POST)"
        }
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Check if the API and model are working properly"""
    try:
        test_tags = tagger.tag_text("This is a test text for health check")
        model_loaded = len(test_tags) > 0
    except:
        model_loaded = False

    return HealthResponse(
        status="healthy" if model_loaded else "unhealthy",
        model_loaded=model_loaded,
        timestamp=datetime.now()
    )

#only tags
@app.post("/tag", response_model=TagResponse)
async def generate_tags(request: TagRequest):
    """Generate tags from input text"""
    try:
        start_time = time.time()

        if request.include_scores:
            tags_with_scores = tagger.tag_text_with_scores(
                request.text, 
                max_tags=request.max_tags
            )
            tags_with_scores = [(tag, score) for tag, score in tags_with_scores 
                               if score >= request.min_score]

            tags = [tag for tag, _ in tags_with_scores]
            scores = [float(score) for _, score in tags_with_scores]  # Convert numpy to float
        else:
            tags_with_scores = tagger.generate_tags(
                request.text,
                max_tags=request.max_tags,
                min_score=request.min_score
            )
            tags = [tag for tag, _ in tags_with_scores]
            scores = None

        processing_time = time.time() - start_time

        return TagResponse(
            tags=tags,
            scores=scores,
            processing_time=processing_time,
            tag_count=len(tags),
            timestamp=datetime.now()
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating tags: {str(e)}")


#tags with details
@app.post("/tag/detailed", response_model=TagWithScoreResponse)
async def generate_tags_detailed(request: TagRequest):
    """Generate tags with detailed information including individual scores and sources"""
    try:
        start_time = time.time()

        # Use the new method that returns sources
        tags_with_sources = tagger.generate_tags_with_source(
            request.text,
            max_tags=request.max_tags,
            min_score=request.min_score
        )

        # Format tags with source information
        tags_formatted = [
            {
                "tag": tag,
                "score": round(float(score), 3),
                "source": source
            }
            for tag, score, source in tags_with_sources
        ]

        processing_time = time.time() - start_time

        return TagWithScoreResponse(
            tags=tags_formatted,
            processing_time=processing_time,
            tag_count=len(tags_formatted),
            timestamp=datetime.now()
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating tags: {str(e)}")

#test
@app.get("/debug/check-clustering")
async def check_clustering():
    test_text = "Apple unveiled its latest M3 chip"

    # Get tags with clustering
    tags_with = tagger.generate_tags(test_text, use_semantic_clustering=True)

    # Get tags without clustering
    tags_without = tagger.generate_tags(test_text, use_semantic_clustering=False)

    return {
        "with_clustering": tags_with,
        "without_clustering": tags_without,
        "clustering_enabled": True
    }






# For Hugging Face Spaces
if __name__ == "__main__":
    print("Starting Uvicorn server...", flush=True)
    uvicorn.run(app, host="0.0.0.0", port=7860)