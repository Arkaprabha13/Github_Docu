from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from utils_pgvector import generate_human_level_repo_docs, HumanLevelRepoAnalyzer
import os
from fastapi.middleware.cors import CORSMiddleware


app = FastAPI(title="Human-Level Repo Documentation with PostgreSQL + pgvector")
# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your frontend domains
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)
class DocRequest(BaseModel):
    repo_url: str
    user_needs: str = ""

@app.get("/health")
async def health_check():
    """Enhanced health check with proper error handling"""
    try:
        # Check environment variables first
        db_url = os.getenv("DATABASE_URL")
        if not db_url:
            return {
                "status": "unhealthy", 
                "error": "DATABASE_URL environment variable not set",
                "suggestion": "Check your .env file"
            }
        
        # Test database connection
        analyzer = HumanLevelRepoAnalyzer()
        
        # Test basic connection
        cursor = analyzer.conn.cursor()
        cursor.execute("SELECT 1 as test, version() as db_version")
        result = cursor.fetchone()
        
        # Check pgvector
        cursor.execute("SELECT * FROM pg_extension WHERE extname = 'vector'")
        pgvector_installed = cursor.fetchone() is not None
        
        analyzer.conn.close()
        
        return {
            "status": "healthy",
            "database_connection": "successful",
            "database_version": result['db_version'][:50] + "...",
            "pgvector_available": pgvector_installed,
            "test_query_result": result['test']
        }
        
    except Exception as e:
        # Detailed error reporting
        error_details = {
            "status": "unhealthy",
            "error_type": type(e).__name__,
            "error_message": str(e),
            "database_url_set": bool(os.getenv("DATABASE_URL")),
        }
        
        # Add specific guidance based on error
        if "DATABASE_URL" in str(e):
            error_details["suggestion"] = "Check DATABASE_URL environment variable"
        elif "connection" in str(e).lower():
            error_details["suggestion"] = "Database connection issue - verify credentials"
        else:
            error_details["suggestion"] = "Check server logs for detailed error information"
            
        return error_details

@app.post("/generate_human_level_docs")
async def generate_human_level_docs(request: DocRequest):
    try:
        result = generate_human_level_repo_docs(request.repo_url, request.user_needs)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))
