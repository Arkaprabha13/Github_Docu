-- Enable pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Create tables for storing code analysis
CREATE TABLE repositories (
    id SERIAL PRIMARY KEY,
    repo_url TEXT UNIQUE NOT NULL,
    repo_name TEXT NOT NULL,
    analyzed_at TIMESTAMP DEFAULT NOW(),
    total_files INTEGER,
    total_lines INTEGER,
    languages JSONB,
    project_type TEXT,
    main_purpose TEXT
);

CREATE TABLE code_files (
    id SERIAL PRIMARY KEY,
    repo_id INTEGER REFERENCES repositories(id),
    file_path TEXT NOT NULL,
    file_type TEXT,
    language TEXT,
    content TEXT,
    lines INTEGER,
    functions JSONB,
    classes JSONB,
    imports JSONB,
    complexity_score INTEGER,
    embedding vector(384),
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE TABLE project_insights (
    id SERIAL PRIMARY KEY,
    repo_id INTEGER REFERENCES repositories(id),
    insight_type TEXT, -- 'pattern', 'todo', 'improvement', 'purpose'
    title TEXT,
    description TEXT,
    confidence_score FLOAT,
    file_references JSONB,
    created_at TIMESTAMP DEFAULT NOW()
);

-- Create indexes for better performance
CREATE INDEX idx_code_files_embedding ON code_files USING ivfflat (embedding vector_cosine_ops);
CREATE INDEX idx_code_files_repo ON code_files(repo_id);
CREATE INDEX idx_project_insights_repo ON project_insights(repo_id);
