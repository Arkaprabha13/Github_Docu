# DocuGen Backend API - Comprehensive Documentation

A production-ready FastAPI backend service that provides AI-powered documentation generation for GitHub repositories using PostgreSQL with pgvector for semantic search and intelligent code analysis.

## 🎯 Project Overview

### What Problem Does This Solve?

**Traditional repository documentation challenges:**
- Manual documentation is time-consuming and often outdated
- Developers spend 20-30% of their time understanding unfamiliar codebases
- Code reviews lack context about architectural decisions
- New team members need weeks to understand project structure
- Technical debt accumulates without proper documentation

**Our Solution:**
DocuGen automatically generates comprehensive, human-level documentation by:
- **Semantic Code Analysis**: Understanding code relationships and patterns
- **AI-Powered Insights**: Generating architectural explanations and best practices
- **Intelligent Context**: Providing setup guides, usage examples, and improvement suggestions
- **Vector-Based Search**: Finding relevant code sections for specific documentation needs

### Key Capabilities

- **Multi-Language Support**: Python, JavaScript, TypeScript, Java, C++, C#, Go, Rust, PHP, Ruby
- **Intelligent File Processing**: Smart detection of project structure, dependencies, and patterns
- **Semantic Search**: Vector-based similarity search for code components
- **Technical Debt Analysis**: Identifies complexity hotspots and improvement opportunities
- **Architectural Pattern Detection**: Recognizes MVC, microservices, component-based patterns
- **Human-Level Documentation**: Generates comprehensive 10-section documentation

## 🏗️ Architecture Deep Dive

### System Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   GitHub Repo   │───▶│  FastAPI Server  │───▶│  PostgreSQL +   │
│                 │    │                  │    │    pgvector     │
└─────────────────┘    │  ┌─────────────┐ │    └─────────────────┘
                       │  │ Git Clone   │ │              │
                       │  │ File Parser │ │              │
                       │  │ AST Analysis│ │              │
                       │  │ Embeddings  │ │              │
                       │  │ LLM Gen     │ │              │
                       │  └─────────────┘ │              │
                       └──────────────────┘              │
                                │                        │
                       ┌────────▼────────┐               │
                       │   Groq API      │               │
                       │ (LLM Processing)│               │
                       └─────────────────┘               │
                                                         │
                       ┌─────────────────────────────────▼─┐
                       │        Vector Similarity Search   │
                       │     Semantic Code Relationships   │
                       └───────────────────────────────────┘
```

### Core Components

#### 1. **Repository Analyzer (`HumanLevelRepoAnalyzer`)**
- **Git Integration**: Shallow cloning with cleanup management
- **File Detection**: Intelligent filtering of relevant code files
- **Multi-Format Support**: Handles Jupyter notebooks, configuration files, documentation
- **Metadata Extraction**: AST parsing for functions, classes, imports, complexity analysis

#### 2. **Vector Processing Pipeline**
- **Sentence Transformers**: Uses `all-MiniLM-L6-v2` for 384-dimensional embeddings
- **Content Optimization**: Truncates large files, preserves structure
- **Batch Processing**: Efficient embedding generation for multiple files
- **Context Building**: Combines file path, language, and content for semantic representation

#### 3. **Database Layer**
- **Schema Management**: Automatic table creation and migration
- **Vector Storage**: Efficient storage of 384-dimensional embeddings
- **Relationship Modeling**: Repository → Files → Insights hierarchy
- **Index Optimization**: IVFFlat indexing for fast similarity search

#### 4. **LLM Integration**
- **Groq API**: High-performance inference using DeepSeek R1 model
- **Prompt Engineering**: Structured prompts for consistent documentation format
- **Context Management**: Smart truncation to fit token limits
- **Fallback Handling**: Local documentation generation when API fails

## 🔄 Why PostgreSQL + pgvector Over Alternatives?

### Vector Database Comparison

| Feature | PostgreSQL + pgvector | FAISS | Qdrant | ChromaDB |
|---------|----------------------|-------|--------|----------|
| **Production Readiness** | ✅ Enterprise-grade | ⚠️ Research-focused | ✅ Good | ⚠️ Emerging |
| **ACID Compliance** | ✅ Full ACID | ❌ No | ✅ Limited | ❌ No |
| **Deployment Complexity** | ✅ Standard PostgreSQL | ❌ Custom setup | ⚠️ Additional service | ⚠️ Additional service |
| **Backup/Recovery** | ✅ Standard tools | ❌ Custom solutions | ⚠️ Custom | ⚠️ Custom |
| **Query Flexibility** | ✅ Full SQL + vectors | ❌ Limited | ⚠️ Custom API | ⚠️ Limited |
| **Free Tier Compatibility** | ✅ Render/Supabase | ❌ No | ❌ No | ❌ No |
| **Metadata Filtering** | ✅ Complex SQL queries | ⚠️ Limited | ✅ Good | ⚠️ Basic |
| **Integration Complexity** | ✅ Single database | ❌ Dual system | ❌ Dual system | ❌ Dual system |

### Specific Reasons for Our Choice

#### **1. Deployment Simplicity**
```python
# PostgreSQL + pgvector: Single database
DATABASE_URL = "postgresql://user:pass@host/db"

# vs. Multiple services approach:
POSTGRES_URL = "postgresql://user:pass@host/db"  # Metadata
QDRANT_URL = "http://qdrant:6333"               # Vectors
REDIS_URL = "redis://redis:6379"                # Cache
```

#### **2. Query Power**
```sql
-- Complex queries with pgvector
SELECT cf.file_path, cf.language, r.repo_name,
       1 - (cf.embedding  %s::vector) AS similarity
FROM code_files cf
JOIN repositories r ON cf.repo_id = r.id
WHERE r.project_type = 'Python Application'
  AND cf.language = 'Python'
  AND cf.complexity_score > 100
ORDER BY similarity DESC
LIMIT 10;
```

#### **3. Free Tier Compatibility**
- **Render**: Managed PostgreSQL with pgvector support
- **Supabase**: Built-in pgvector, generous free tier
- **Railway**: PostgreSQL with extensions
- **No additional vector service costs**

#### **4. Production Reliability**
- **ACID Transactions**: Consistent state during analysis
- **Proven Scalability**: PostgreSQL handles enterprise workloads
- **Standard Tooling**: Existing monitoring, backup, security tools
- **Team Familiarity**: Most developers know SQL

#### **5. Cost Efficiency**
```
Traditional Vector Setup:
- PostgreSQL (metadata): $20/month
- Qdrant/Pinecone (vectors): $70/month
- Total: $90/month

Our Approach:
- PostgreSQL + pgvector: $20/month
- Total: $20/month (70% cost reduction)
```

## 🚀 Comprehensive Installation Guide

### Prerequisites Verification

#### System Requirements
```bash
# Check Python version (3.8+ required)
python --version

# Check PostgreSQL installation
psql --version

# Verify git installation
git --version

# Check available memory (2GB+ recommended)
free -h  # Linux
top      # macOS
```

#### Hardware Requirements
- **Minimum**: 2GB RAM, 1 CPU core, 5GB storage
- **Recommended**: 4GB RAM, 2 CPU cores, 20GB storage
- **Production**: 8GB RAM, 4 CPU cores, 100GB storage

### Detailed Setup Process

#### 1. Environment Setup

```bash
# Clone repository
git clone https://github.com/your-username/docugen-backend.git
cd docugen-backend

# Create isolated Python environment
python -m venv docugen_env

# Activate environment
# Windows:
docugen_env\Scripts\activate
# macOS/Linux:
source docugen_env/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install dependencies with specific versions
pip install -r requirements.txt
```

#### 2. PostgreSQL with pgvector Setup

##### Option A: Local Installation

**Ubuntu/Debian:**
```bash
# Install PostgreSQL
sudo apt update
sudo apt install postgresql postgresql-contrib postgresql-client

# Install build tools for pgvector
sudo apt install build-essential postgresql-server-dev-15

# Install pgvector
git clone --branch v0.5.1 https://github.com/pgvector/pgvector.git
cd pgvector
make
sudo make install

# Start PostgreSQL
sudo systemctl start postgresql
sudo systemctl enable postgresql
```

**macOS:**
```bash
# Install via Homebrew
brew install postgresql@15
brew install pgvector

# Start PostgreSQL
brew services start postgresql@15
```

**Windows:**
```bash
# Download and install PostgreSQL from postgresql.org
# Download pgvector Windows binary from releases page
# Or use Docker (recommended for Windows)
```

##### Option B: Docker Setup

```yaml
# docker-compose.yml
version: '3.8'
services:
  postgres:
    image: pgvector/pgvector:pg15
    environment:
      POSTGRES_DB: docugen_db
      POSTGRES_USER: docugen_user
      POSTGRES_PASSWORD: secure_password
    ports:
      - "5432:5432"
    volumes:
      - postgres_data:/var/lib/postgresql/data
      - ./init.sql:/docker-entrypoint-initdb.d/init.sql

volumes:
  postgres_data:
```

```bash
# Start with Docker
docker-compose up -d
```

##### Option C: Cloud Setup (Recommended)

**Supabase (Free Tier):**
```bash
# 1. Create account at supabase.com
# 2. Create new project
# 3. Go to SQL editor and run:
CREATE EXTENSION IF NOT EXISTS vector;

# 4. Get connection string from Settings > Database
```

**Render (Free Tier):**
```bash
# 1. Create account at render.com
# 2. Create PostgreSQL service
# 3. Connect and enable extension:
CREATE EXTENSION IF NOT EXISTS vector;
```

#### 3. Database Schema Setup

```sql
-- Connect to your database
psql "postgresql://user:password@host:port/database"

-- Enable pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Create optimized schema
CREATE TABLE repositories (
    id SERIAL PRIMARY KEY,
    repo_url TEXT UNIQUE NOT NULL,
    repo_name TEXT NOT NULL,
    analyzed_at TIMESTAMP DEFAULT NOW(),
    total_files INTEGER CHECK (total_files >= 0),
    total_lines INTEGER CHECK (total_lines >= 0),
    languages JSONB DEFAULT '{}',
    project_type TEXT DEFAULT 'Unknown',
    main_purpose TEXT DEFAULT 'General Software Development',
    analysis_duration FLOAT,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

CREATE TABLE code_files (
    id SERIAL PRIMARY KEY,
    repo_id INTEGER REFERENCES repositories(id) ON DELETE CASCADE,
    file_path TEXT NOT NULL,
    file_type TEXT,
    language TEXT DEFAULT 'Unknown',
    content TEXT,
    lines INTEGER CHECK (lines >= 0),
    functions JSONB DEFAULT '[]',
    classes JSONB DEFAULT '[]',
    imports JSONB DEFAULT '[]',
    todos JSONB DEFAULT '[]',
    complexity_score INTEGER CHECK (complexity_score >= 0),
    embedding vector(384),
    created_at TIMESTAMP DEFAULT NOW(),
    UNIQUE(repo_id, file_path)
);

CREATE TABLE project_insights (
    id SERIAL PRIMARY KEY,
    repo_id INTEGER REFERENCES repositories(id) ON DELETE CASCADE,
    insight_type TEXT NOT NULL,
    title TEXT NOT NULL,
    description TEXT,
    confidence_score FLOAT CHECK (confidence_score >= 0 AND confidence_score  query_embedding) AS similarity
    FROM code_files cf
    WHERE 
        cf.repo_id = target_repo_id
        AND cf.embedding IS NOT NULL
        AND 1 - (cf.embedding  query_embedding) > match_threshold
    ORDER BY cf.embedding  query_embedding
    LIMIT match_count;
$$;
```

#### 4. Environment Configuration

```bash
# Create .env file
cat > .env  '[0.1,0.2,...]'::vector 
LIMIT 10;
```

#### Caching Strategy
```python
# Add Redis caching for frequent queries
import redis
from functools import wraps

redis_client = redis.Redis(host='localhost', port=6379, db=0)

def cache_result(expiration=3600):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            cache_key = f"{func.__name__}:{hash(str(args) + str(kwargs))}"
            cached = redis_client.get(cache_key)
            if cached:
                return json.loads(cached)
            
            result = func(*args, **kwargs)
            redis_client.setex(cache_key, expiration, json.dumps(result))
            return result
        return wrapper
    return decorator
```

### Production Deployment

#### Docker Configuration
```dockerfile
# Dockerfile
FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create non-root user
RUN useradd -m -u 1000 docugen && chown -R docugen:docugen /app
USER docugen

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:8000/health || exit 1

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

#### Kubernetes Deployment
```yaml
# k8s/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: docugen-backend
spec:
  replicas: 3
  selector:
    matchLabels:
      app: docugen-backend
  template:
    metadata:
      labels:
        app: docugen-backend
    spec:
      containers:
      - name: docugen-backend
        image: docugen/backend:latest
        ports:
        - containerPort: 8000
        env:
        - name: DATABASE_URL
          valueFrom:
            secretKeyRef:
              name: docugen-secrets
              key: database-url
        - name: GROQ_API_KEY
          valueFrom:
            secretKeyRef:
              name: docugen-secrets
              key: groq-api-key
        resources:
          requests:
            memory: "1Gi"
            cpu: "500m"
          limits:
            memory: "2Gi"
            cpu: "1000m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 5
```

## 🚨 Comprehensive Troubleshooting

### Common Issues and Solutions

#### 1. Database Connection Issues

**Problem**: `psycopg2.OperationalError: could not connect to server`

**Diagnosis:**
```bash
# Test basic connectivity
telnet localhost 5432

# Check PostgreSQL status
sudo systemctl status postgresql

# Verify user permissions
sudo -u postgres psql -c "\du"
```

**Solutions:**
```bash
# Fix authentication
sudo nano /etc/postgresql/15/main/pg_hba.conf
# Change: local all all peer
# To:     local all all md5

# Restart PostgreSQL
sudo systemctl restart postgresql

# Update connection string format
DATABASE_URL="postgresql://user:password@localhost:5432/database?sslmode=disable"
```

#### 2. pgvector Extension Issues

**Problem**: `extension "vector" is not available`

**Diagnosis:**
```sql
-- Check available extensions
SELECT name FROM pg_available_extensions WHERE name LIKE '%vector%';

-- Check installed extensions
SELECT extname FROM pg_extension;
```

**Solutions:**
```bash
# Reinstall pgvector
sudo apt remove postgresql-15-pgvector
sudo apt install postgresql-15-pgvector

# Manual installation
git clone https://github.com/pgvector/pgvector.git
cd pgvector
make clean
make OPTFLAGS=""
sudo make install

# Restart PostgreSQL and reconnect
sudo systemctl restart postgresql
psql -c "CREATE EXTENSION vector;"
```

#### 3. Memory and Performance Issues

**Problem**: `Out of memory` or slow embedding generation

**Diagnosis:**
```bash
# Monitor memory usage
htop

# Check Python memory usage
python -c "
import psutil
import os
process = psutil.Process(os.getpid())
print(f'Memory: {process.memory_info().rss / 1024 / 1024:.2f} MB')
"
```

**Solutions:**
```python
# Optimize embedding batch size
EMBEDDING_BATCH_SIZE = 8  # Reduce from 32

# Use CPU-only torch
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Implement memory monitoring
import psutil
def check_memory():
    if psutil.virtual_memory().percent > 80:
        gc.collect()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
```

#### 4. Groq API Issues

**Problem**: Rate limiting or authentication errors

**Diagnosis:**
```python
# Test API key
import requests
headers = {"Authorization": f"Bearer {GROQ_API_KEY}"}
response = requests.get("https://api.groq.com/openai/v1/models", headers=headers)
print(response.status_code, response.text)
```

**Solutions:**
```python
# Implement exponential backoff
import time
import random

def retry_with_backoff(func, max_retries=3):
    for attempt in range(max_retries):
        try:
            return func()
        except Exception as e:
            if "rate_limit" in str(e).lower():
                wait_time = (2 ** attempt) + random.uniform(0, 1)
                time.sleep(wait_time)
                continue
            raise e
    raise Exception("Max retries exceeded")
```

#### 5. Git Clone Issues

**Problem**: Repository access denied or timeout

**Diagnosis:**
```bash
# Test git access
git clone --depth 1 https://github.com/user/repo.git /tmp/test
```

**Solutions:**
```python
# Add timeout and retry logic
from git import Repo
import subprocess

def safe_clone(repo_url, target_dir, timeout=60):
    try:
        # Use subprocess with timeout
        result = subprocess.run([
            'git', 'clone', '--depth', '1', 
            '--single-branch', repo_url, target_dir
        ], timeout=timeout, capture_output=True, text=True)
        
        if result.returncode != 0:
            raise Exception(f"Git clone failed: {result.stderr}")
        return target_dir
    except subprocess.TimeoutExpired:
        raise Exception("Repository clone timeout")
```

### Performance Monitoring

#### Database Performance
```sql
-- Monitor query performance
SELECT query, mean_exec_time, calls, total_exec_time
FROM pg_stat_statements
WHERE query LIKE '%embedding%'
ORDER BY mean_exec_time DESC;

-- Monitor index usage
SELECT schemaname, tablename, indexname, idx_scan, idx_tup_read
FROM pg_stat_user_indexes
WHERE indexname LIKE '%embedding%';

-- Check vector index effectiveness
EXPLAIN (ANALYZE, BUFFERS) 
SELECT file_path, 1 - (embedding  $1) as similarity
FROM code_files 
ORDER BY embedding  $1 
LIMIT 10;
```

#### Application Monitoring
```python
# Add timing decorators
import functools
import time
import logging

def monitor_performance(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        try:
            result = func(*args, **kwargs)
            execution_time = time.time() - start_time
            logging.info(f"{func.__name__} executed in {execution_time:.2f}s")
            return result
        except Exception as e:
            execution_time = time.time() - start_time
            logging.error(f"{func.__name__} failed after {execution_time:.2f}s: {e}")
            raise
    return wrapper
```

## 📊 Performance Benchmarks

### Expected Performance Metrics

| Repository Size | Processing Time | Memory Usage | Database Size |
|----------------|-----------------|--------------|---------------|
| Small (-
          --health-cmd pg_isready
          --health-interval 10s
          --health-timeout 5s
          --health-retries 5
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python
      uses: actions/setup-python@v3
      with:
        python-version: '3.11'
    
    - name: Install dependencies
      run: |
        pip install -r requirements.txt
        pip install -r requirements-dev.txt
    
    - name: Run tests
      env:
        DATABASE_URL: postgresql://postgres:postgres@localhost:5432/test_db
        GROQ_API_KEY: ${{ secrets.GROQ_API_KEY }}
      run: |
        pytest tests/ -v --cov=app --cov-report=xml
    
    - name: Upload coverage
      uses: codecov/codecov-action@v3
```

This comprehensive documentation provides everything needed to understand, deploy, and maintain the DocuGen backend system. The choice of PostgreSQL + pgvector provides the optimal balance of functionality, reliability, and cost-effectiveness for this use case.
