import os
import logging
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

# Load environment variables FIRST (critical for Render)
load_dotenv()

# Now import your modules after env vars are loaded
from utils_pgvector import generate_human_level_repo_docs, HumanLevelRepoAnalyzer

# Setup logging for better debugging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Human-Level Repository Documentation API",
    description="AI-powered repository analysis and documentation generation using PostgreSQL + pgvector",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        # "http://localhost:3000",
        # "http://localhost:5173",
        # "http://localhost:8080",
        # "https://your-frontend-domain.com",  # Replace with your actual frontend domain
        "*"  # Remove this in production for security
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

class DocRequest(BaseModel):
    repo_url: str
    user_needs: str = "Generate comprehensive documentation"

# Global analyzer instance (initialized once)
_analyzer_instance = None

def get_analyzer():
    """Get or create analyzer instance"""
    global _analyzer_instance
    if _analyzer_instance is None:
        try:
            _analyzer_instance = HumanLevelRepoAnalyzer()
            logger.info("✅ HumanLevelRepoAnalyzer initialized successfully")
        except Exception as e:
            logger.error(f"❌ Failed to initialize analyzer: {e}")
            raise
    return _analyzer_instance

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Repository Documentation API is running",
        "status": "healthy",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "generate_docs": "/generate_human_level_docs",
            "api_docs": "/docs",
            "redoc": "/redoc"
        },
        "features": [
            "AI-powered code analysis",
            "PostgreSQL + pgvector integration",
            "Human-level documentation generation",
            "Multi-language repository support"
        ]
    }

@app.get("/health")
async def health_check():
    """Enhanced health check with comprehensive diagnostics"""
    try:
        logger.info("🔍 Starting health check...")
        
        # Check environment variables
        db_url = os.getenv("DATABASE_URL")
        groq_key = os.getenv("GROQ_API_KEY")
        
        if not db_url:
            return {
                "status": "unhealthy", 
                "error": "DATABASE_URL environment variable not set",
                "suggestion": "Configure DATABASE_URL in Render environment variables"
            }
        
        if not groq_key:
            logger.warning("⚠️ GROQ_API_KEY not set - LLM features will be limited")
        
        # Test database connection using a lightweight approach
        try:
            # Create a test connection without full analyzer initialization
            import psycopg2
            from psycopg2.extras import RealDictCursor
            
            conn = psycopg2.connect(
                db_url,
                cursor_factory=RealDictCursor,
                sslmode='require',
                connect_timeout=10
            )
            
            cursor = conn.cursor()
            cursor.execute("SELECT 1 as test, version() as db_version, current_database() as db_name")
            result = cursor.fetchone()
            
            # Check pgvector extension
            cursor.execute("SELECT EXISTS(SELECT 1 FROM pg_extension WHERE extname = 'vector') as pgvector_available")
            pgvector_result = cursor.fetchone()
            
            conn.close()
            
            logger.info("✅ Health check completed successfully")
            
            return {
                "status": "healthy",
                "database_connection": "successful",
                "database_name": result['db_name'],
                "database_version": result['db_version'][:60] + "...",
                "pgvector_available": pgvector_result['pgvector_available'],
                "test_query_result": result['test'],
                "groq_api_configured": bool(groq_key),
                "environment": os.getenv("RENDER", "local") and "render" or "local",
                "timestamp": "2025-07-19T18:00:00Z"
            }
            
        except psycopg2.OperationalError as db_error:
            logger.error(f"❌ Database connection failed: {db_error}")
            return {
                "status": "unhealthy",
                "error_type": "DatabaseConnectionError",
                "error_message": str(db_error),
                "suggestion": "Check DATABASE_URL format and database availability"
            }
        
    except Exception as e:
        logger.error(f"❌ Health check failed: {e}")
        return {
            "status": "unhealthy",
            "error_type": type(e).__name__,
            "error_message": str(e),
            "database_url_configured": bool(os.getenv("DATABASE_URL")),
            "suggestion": "Check server logs for detailed error information"
        }

@app.post("/generate_human_level_docs")
async def generate_docs(request: DocRequest):
    """Generate comprehensive repository documentation"""
    try:
        logger.info(f"📋 Starting documentation generation for: {request.repo_url}")
        
        # Validate repository URL
        if not request.repo_url.strip():
            raise HTTPException(status_code=400, detail="Repository URL is required")
        
        if not request.repo_url.startswith(("https://github.com/", "http://github.com/")):
            raise HTTPException(
                status_code=400, 
                detail="Please provide a valid GitHub repository URL (https://github.com/username/repository)"
            )
        
        # Generate documentation
        result = generate_human_level_repo_docs(request.repo_url, request.user_needs)
        
        logger.info(f"✅ Documentation generation completed for: {request.repo_url}")
        return result
        
    except HTTPException:
        # Re-raise HTTP exceptions as-is
        raise
    except Exception as e:
        logger.error(f"❌ Documentation generation failed: {e}")
        raise HTTPException(
            status_code=500, 
            detail={
                "error": "Documentation generation failed",
                "message": str(e),
                "type": type(e).__name__,
                "suggestion": "Please try again or contact support if the issue persists"
            }
        )

# Startup event to verify system readiness
@app.on_event("startup")
async def startup_event():
    """Initialize system on startup"""
    logger.info("🚀 Starting Repository Documentation API...")
    
    # Verify critical environment variables
    required_vars = ["DATABASE_URL"]
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    
    if missing_vars:
        logger.error(f"❌ Missing required environment variables: {missing_vars}")
        raise Exception(f"Missing environment variables: {missing_vars}")
    
    # Optional: Pre-initialize analyzer to catch startup errors early
    try:
        get_analyzer()
        logger.info("✅ System initialization completed successfully")
    except Exception as e:
        logger.error(f"❌ System initialization failed: {e}")
        # Don't raise here to allow health checks to still work

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("🔄 Shutting down Repository Documentation API...")
    
    # Cleanup analyzer if needed
    global _analyzer_instance
    if _analyzer_instance:
        try:
            _analyzer_instance.cleanup_resources()
            logger.info("✅ Resources cleaned up successfully")
        except Exception as e:
            logger.warning(f"⚠️ Cleanup warning: {e}")

# Production server configuration
if __name__ == "__main__":
    import uvicorn
    
    # Get port from environment (Render sets this)
    port = int(os.environ.get("PORT", 8000))
    
    # Log startup information
    print(f"🚀 Starting Repository Documentation API")
    print(f"📍 Server: 0.0.0.0:{port}")
    print(f"🌍 Environment: {os.getenv('RENDER', 'local') and 'Render' or 'Local'}")
    print(f"📊 Database: {'✅ Configured' if os.getenv('DATABASE_URL') else '❌ Not configured'}")
    print(f"🤖 Groq API: {'✅ Configured' if os.getenv('GROQ_API_KEY') else '❌ Not configured'}")
    
    # Run server
    uvicorn.run(
        app,
        host="0.0.0.0",  # Critical: Must bind to all interfaces for Render
        port=port,       # Use Render's PORT environment variable
        log_level="info",
        access_log=True
    )
