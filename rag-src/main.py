import logging
from common.service.configloader import settings, deep_get
from common.service.logging_setup import setup_logging

from fastapi import FastAPI
#from fastapi.staticfiles import StaticFiles
from api import retrieval_search_api_endpoints_main
#from api import rag_chat_endpoints
#from api import admin_api_endpoints
from index_builder_and_retrieval_search_service import build_index

logger = logging.getLogger(__name__)

# setup
setup_logging()


# start building the index
build_index.start_indexing()

# start server with API
app = FastAPI()

@app.get("/hello")
def get_hello():
    return {"message": "Hello from the AI RAG API"}

@app.get("/health")
# https://stackoverflow.com/questions/46949108/spec-for-http-health-checks/47119512#47119512
def get_health():
    return {"status": "healthy", "message": "AI RAG API is running"}

# /search/*  (SearxNG API compatible)
app.include_router(retrieval_search_api_endpoints_main.router)

# /api/*     (Ollama / Open AI compatible)
#app.include_router(rag_chat_endpoints.router)

# /admin/*
#app.include_router(admin_api_endpoints.router)

# /* for static files - the request are processed IN THE ORDER the mounts are defined here
#app.mount("/chat", StaticFiles(directory="static_chat",html = True))
#app.mount("/", StaticFiles(directory="static",html = True))

# run the HTTP server
if __name__ == "__main__":
    import uvicorn
    logger.info("Starting FastAPI server...")
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=8000,
        log_level="info",
        access_log=True
    )
