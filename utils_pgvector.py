import os
import shutil
import tempfile
import psycopg2
from psycopg2.extras import RealDictCursor, Json
import numpy as np
from git import Repo as GitRepo
from groq import Groq
from dotenv import load_dotenv
from typing import Dict, List, Any, Tuple, Optional
import json
from pathlib import Path
import hashlib
from dataclasses import dataclass
from sentence_transformers import SentenceTransformer
from datetime import datetime
import stat
import re
import ast
from urllib.parse import urlparse
import traceback
import logging
import sys

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Verify critical environment variables exist
if not os.getenv("DATABASE_URL"):
    raise Exception("DATABASE_URL environment variable is required")
if not os.getenv("GROQ_API_KEY"):
    logger.warning("⚠️ GROQ_API_KEY not set - LLM features will fail")

@dataclass
class CodeFile:
    path: str
    content: str
    file_type: str
    language: str
    size: int
    lines: int
    functions: List[Dict]
    classes: List[Dict]
    imports: List[str]
    todos: List[str]
    complexity_score: int

@dataclass
class ProjectInsight:
    insight_type: str
    title: str
    description: str
    confidence_score: float
    file_references: List[str]

class HumanLevelRepoAnalyzer:
    def __init__(self):
        """Initialize the repository analyzer with comprehensive setup"""
        logger.info("🚀 Initializing HumanLevelRepoAnalyzer")
        
        # Database connection for PostgreSQL
        self.db_url = os.getenv("DATABASE_URL")
        self.conn = None
        
        # Supported file extensions and their languages
        self.supported_extensions = {
            # Web Technologies
            '.js': 'JavaScript', '.jsx': 'JavaScript/React', '.ts': 'TypeScript', 
            '.tsx': 'TypeScript/React', '.html': 'HTML', '.css': 'CSS', 
            '.scss': 'SASS', '.vue': 'Vue.js', '.less': 'LESS',
            
            # Backend Languages
            '.py': 'Python', '.java': 'Java', '.cpp': 'C++', '.c': 'C',
            '.cs': 'C#', '.go': 'Go', '.rs': 'Rust', '.php': 'PHP',
            '.rb': 'Ruby', '.swift': 'Swift', '.kt': 'Kotlin',
            
            # Data & Config
            '.json': 'JSON', '.yaml': 'YAML', '.yml': 'YAML', 
            '.toml': 'TOML', '.env': 'Environment', '.xml': 'XML',
            '.ini': 'INI', '.cfg': 'Config',
            
            # Notebooks & Scripts
            '.ipynb': 'Jupyter Notebook', '.md': 'Markdown', '.txt': 'Text',
            '.bat': 'Batch Script', '.sh': 'Shell Script', '.ps1': 'PowerShell',
            '.rst': 'reStructuredText',
            
            # Build & Package files
            'requirements.txt': 'Python Requirements', 'package.json': 'NPM Package',
            'dockerfile': 'Docker', 'docker-compose.yml': 'Docker Compose',
            'pom.xml': 'Maven', 'build.gradle': 'Gradle', 'cargo.toml': 'Rust Package',
            'makefile': 'Makefile', 'cmakelists.txt': 'CMake',
            '.gitignore': 'Git Ignore', '.env.example': 'Environment Template'
        }
        
        # Initialize components
        self.embedding_model = None
        self.temp_directories = []
        
        try:
            # Initialize embedding model with error handling
            self.initialize_embedding_model()
            
            # Connect to database
            self.connect_db()
            
            # Ensure database tables exist
            self.setup_database_tables()
            
            logger.info("✅ HumanLevelRepoAnalyzer initialized successfully")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize analyzer: {e}")
            self.cleanup_resources()
            raise

    def initialize_embedding_model(self):
        """Initialize sentence transformer model with proper error handling"""
        try:
            logger.info("🔄 Loading sentence transformer model...")
            
            # Create cache directory if it doesn't exist
            cache_dir = os.path.expanduser("~/.cache/torch/sentence_transformers")
            os.makedirs(cache_dir, exist_ok=True)
            
            # Initialize with explicit device and cache settings
            self.embedding_model = SentenceTransformer(
                'all-MiniLM-L6-v2',
                cache_folder=cache_dir,
                device='cpu'  # Force CPU to avoid CUDA issues
            )
            
            # Test the model
            test_embedding = self.embedding_model.encode(["Test sentence"])
            if len(test_embedding[0]) != 384:
                raise ValueError(f"Expected 384 dimensions, got {len(test_embedding[0])}")
            
            logger.info("✅ Embedding model loaded successfully")
            
        except Exception as e:
            logger.error(f"❌ Failed to load embedding model: {e}")
            raise Exception(f"Embedding model initialization failed: {str(e)}")

    def connect_db(self):
        """Connect to PostgreSQL database with comprehensive error handling"""
        try:
            if not self.db_url:
                raise Exception("DATABASE_URL environment variable not set")
            
            logger.info("🔄 Connecting to PostgreSQL database...")
            
            # Parse URL safely for debugging
            try:
                parsed = urlparse(self.db_url)
                logger.info(f"🔗 Database host: {parsed.hostname}:{parsed.port or 5432}")
                logger.info(f"🔗 Database: {parsed.path.lstrip('/')}")
                logger.info(f"🔗 Username: {parsed.username}")
            except Exception as parse_error:
                logger.warning(f"⚠️ Could not parse database URL: {parse_error}")
            
            # Connect with explicit error handling
            self.conn = psycopg2.connect(
                self.db_url,
                cursor_factory=RealDictCursor,
                sslmode='require',
                connect_timeout=30,
                application_name='RepoAnalyzer'
            )
            self.conn.autocommit = False
            
            # Verify connection with comprehensive tests
            cursor = self.conn.cursor()
            cursor.execute("SELECT version(), current_user, current_database()")
            version, user, database = cursor.fetchone()
            
            logger.info(f"✅ Connected as {user} to database {database}")
            logger.info(f"✅ PostgreSQL: {version[:60]}...")
            
            # Check pgvector extension
            cursor.execute("SELECT * FROM pg_extension WHERE extname = 'vector'")
            if cursor.fetchone():
                logger.info("✅ pgvector extension available")
                
                # Test vector operations
                cursor.execute("SELECT '[1,2,3]'::vector as test_vector")
                test_result = cursor.fetchone()
                logger.info(f"✅ Vector operations working: {test_result['test_vector']}")
            else:
                logger.warning("⚠️ pgvector extension not found - enable it in your database")
            
            cursor.close()
            
        except psycopg2.OperationalError as e:
            error_msg = str(e)
            logger.error(f"❌ PostgreSQL Operational Error: {error_msg}")
            if "timeout" in error_msg.lower():
                raise Exception("Database connection timeout - check network connectivity")
            elif "authentication failed" in error_msg.lower():
                raise Exception("Database authentication failed - check credentials")
            else:
                raise Exception(f"Database connection failed: {error_msg}")
        except Exception as e:
            logger.error(f"❌ Database connection error: {e}")
            raise Exception(f"Database connection error: {str(e)}")

    def setup_database_tables(self):
        """Create database tables with comprehensive error handling"""
        try:
            logger.info("🔄 Setting up database tables...")
            cursor = self.conn.cursor()
            
            # Enable pgvector extension
            cursor.execute("CREATE EXTENSION IF NOT EXISTS vector")
            
            # Create repositories table with better constraints
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS repositories (
                    id SERIAL PRIMARY KEY,
                    repo_url TEXT UNIQUE NOT NULL,
                    repo_name TEXT NOT NULL,
                    analyzed_at TIMESTAMP DEFAULT NOW(),
                    total_files INTEGER CHECK (total_files >= 0),
                    total_lines INTEGER CHECK (total_lines >= 0),
                    languages JSONB DEFAULT '{}',
                    project_type TEXT DEFAULT 'Unknown',
                    main_purpose TEXT DEFAULT 'General Software Development'
                )
            """)
            
            # Create code_files table with better constraints
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS code_files (
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
                    complexity_score INTEGER CHECK (complexity_score >= 0),
                    embedding vector(384),
                    created_at TIMESTAMP DEFAULT NOW(),
                    UNIQUE(repo_id, file_path)
                )
            """)
            
            # Create project_insights table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS project_insights (
                    id SERIAL PRIMARY KEY,
                    repo_id INTEGER REFERENCES repositories(id) ON DELETE CASCADE,
                    insight_type TEXT NOT NULL,
                    title TEXT NOT NULL,
                    description TEXT,
                    confidence_score FLOAT CHECK (confidence_score >= 0 AND confidence_score <= 1),
                    file_references JSONB DEFAULT '[]',
                    created_at TIMESTAMP DEFAULT NOW()
                )
            """)
            
            # Create indexes with error handling
            try:
                cursor.execute("""
                    CREATE INDEX IF NOT EXISTS idx_code_files_embedding 
                    ON code_files USING ivfflat (embedding vector_cosine_ops)
                    WITH (lists = 100)
                """)
            except Exception as idx_error:
                logger.warning(f"⚠️ Could not create vector index: {idx_error}")
            
            # Create other indexes
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_code_files_repo ON code_files(repo_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_project_insights_repo ON project_insights(repo_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_repositories_url ON repositories(repo_url)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_code_files_language ON code_files(language)")
            
            self.conn.commit()
            cursor.close()
            logger.info("✅ Database tables setup completed")
            
        except Exception as e:
            if self.conn:
                self.conn.rollback()
            logger.error(f"❌ Database setup error: {e}")
            raise

    def clone_and_analyze_repo(self, repo_url: str) -> Tuple[int, List[CodeFile]]:
        """Clone repository and perform human-level analysis with comprehensive error handling"""
        temp_dir = None
        
        try:
            logger.info(f"🔄 Starting repository analysis: {repo_url}")
            
            # Create temporary directory
            temp_dir = tempfile.mkdtemp(prefix="repo_analysis_")
            self.temp_directories.append(temp_dir)
            logger.info(f"📁 Created temp directory: {temp_dir}")
            
            # Clone repository with error handling
            logger.info(f"🔄 Cloning repository: {repo_url}")
            try:
                repo = GitRepo.clone_from(
                    repo_url, 
                    temp_dir,
                    depth=1,  # Shallow clone for performance
                    single_branch=True
                )
                logger.info(f"✅ Repository cloned successfully")
            except Exception as clone_error:
                if "not found" in str(clone_error).lower():
                    raise ValueError(f"Repository not found or is private: {repo_url}")
                elif "timeout" in str(clone_error).lower():
                    raise ValueError(f"Repository clone timeout: {repo_url}")
                else:
                    raise Exception(f"Failed to clone repository: {clone_error}")
            
            # Extract repo name safely
            try:
                repo_name = urlparse(repo_url).path.strip('/').split('/')[-1]
                if not repo_name:
                    repo_name = "unknown_repo"
            except:
                repo_name = "unknown_repo"
            
            # Analyze files with progress tracking
            logger.info("🔄 Analyzing repository files...")
            code_files = self.analyze_all_files_intelligently(temp_dir)
            
            if not code_files:
                logger.warning("⚠️ No analyzable files found in repository")
                # Create minimal entry
                code_files = [CodeFile(
                    path="README.md",
                    content="Empty repository or no analyzable files found",
                    file_type=".md",
                    language="Markdown",
                    size=0,
                    lines=1,
                    functions=[],
                    classes=[],
                    imports=[],
                    todos=[],
                    complexity_score=1
                )]
            
            # Store in database
            logger.info("🔄 Storing analysis results in database...")
            repo_id = self.store_repository_data(repo_url, repo_name, code_files)
            
            logger.info(f"🧠 Analysis completed: {len(code_files)} files analyzed")
            return repo_id, code_files
            
        except Exception as e:
            logger.error(f"❌ Repository analysis failed: {e}")
            raise
        finally:
            # Clean up temporary directory
            if temp_dir and os.path.exists(temp_dir):
                try:
                    self.safe_remove_tree(temp_dir)
                    if temp_dir in self.temp_directories:
                        self.temp_directories.remove(temp_dir)
                except Exception as cleanup_error:
                    logger.warning(f"⚠️ Cleanup warning for {temp_dir}: {cleanup_error}")

    def safe_remove_tree(self, path: str):
        """Safely remove directory tree with comprehensive error handling"""
        max_retries = 3
        retry_count = 0
        
        while retry_count < max_retries:
            try:
                if not os.path.exists(path):
                    return
                
                # First, try normal removal
                shutil.rmtree(path)
                logger.debug(f"✅ Successfully removed directory: {path}")
                return
                
            except PermissionError:
                retry_count += 1
                logger.warning(f"⚠️ Permission error removing {path}, attempt {retry_count}/{max_retries}")
                
                try:
                    # Handle Windows permission issues
                    for root, dirs, files in os.walk(path):
                        for d in dirs:
                            try:
                                dir_path = os.path.join(root, d)
                                os.chmod(dir_path, stat.S_IRWXU)
                            except:
                                pass
                        for f in files:
                            try:
                                file_path = os.path.join(root, f)
                                os.chmod(file_path, stat.S_IRWXU)
                            except:
                                pass
                    
                    # Try removal again
                    shutil.rmtree(path)
                    logger.debug(f"✅ Successfully removed directory after permission fix: {path}")
                    return
                    
                except Exception as fix_error:
                    if retry_count >= max_retries:
                        logger.error(f"❌ Failed to remove directory {path} after {max_retries} attempts: {fix_error}")
                    else:
                        import time
                        time.sleep(1)  # Wait before retry
            
            except Exception as e:
                logger.error(f"❌ Unexpected error removing directory {path}: {e}")
                break

    def analyze_all_files_intelligently(self, repo_path: str) -> List[CodeFile]:
        """Human-level intelligent analysis of all files with progress tracking"""
        code_files = []
        processed_count = 0
        error_count = 0
        
        try:
            # Get total file count for progress tracking
            total_files = sum(1 for root, dirs, files in os.walk(repo_path) 
                            for file in files if not any(skip in root for skip in ['.git', 'node_modules', '__pycache__']))
            
            logger.info(f"📊 Processing {total_files} files...")
            
            for root, dirs, files in os.walk(repo_path):
                # Skip irrelevant directories
                dirs[:] = [d for d in dirs if not d.startswith(
                    ('.git', 'node_modules', '__pycache__', '.pytest_cache', 
                     'venv', '.venv', 'dist', 'build', 'target', 'bin', 'obj')
                )]
                
                for file in files:
                    try:
                        file_path = os.path.join(root, file)
                        relative_path = os.path.relpath(file_path, repo_path)
                        
                        # Skip files that are too large (> 1MB)
                        if os.path.getsize(file_path) > 1024 * 1024:
                            logger.debug(f"⏭️ Skipping large file: {relative_path}")
                            continue
                        
                        # Process relevant files
                        if self.should_analyze_file(file, relative_path):
                            code_file = self.process_file_intelligently(file_path, relative_path)
                            if code_file:
                                code_files.append(code_file)
                        
                        processed_count += 1
                        
                        # Progress logging
                        if processed_count % 20 == 0:
                            logger.info(f"📈 Progress: {processed_count}/{total_files} files processed")
                        
                    except Exception as e:
                        error_count += 1
                        logger.warning(f"⚠️ Error processing {relative_path}: {e}")
                        
                        # Stop processing if too many errors
                        if error_count > 50:
                            logger.error("❌ Too many file processing errors, stopping analysis")
                            break
            
            logger.info(f"✅ File analysis completed: {len(code_files)} files analyzed, {error_count} errors")
            return code_files
            
        except Exception as e:
            logger.error(f"❌ File analysis failed: {e}")
            raise

    def should_analyze_file(self, filename: str, relative_path: str) -> bool:
        """Intelligent file filtering with improved logic"""
        try:
            file_ext = Path(filename).suffix.lower()
            
            # Skip binary files, images, etc.
            skip_extensions = {
                '.png', '.jpg', '.jpeg', '.gif', '.svg', '.ico', 
                '.woff', '.woff2', '.ttf', '.eot', '.mp4', '.mp3',
                '.zip', '.tar', '.gz', '.exe', '.dll', '.so',
                '.pyc', '.pyo', '.class', '.o', '.obj', '.bin',
                '.pdf', '.doc', '.docx', '.xls', '.xlsx', '.ppt', '.pptx'
            }
            
            if file_ext in skip_extensions:
                return False
            
            # Skip hidden files except important ones
            if filename.startswith('.') and filename not in ['.env', '.gitignore', '.dockerignore']:
                return False
            
            # Skip certain directories
            skip_dirs = ['node_modules', '__pycache__', '.git', 'venv', 'env', 'dist', 'build', 'target']
            if any(skip_dir in relative_path for skip_dir in skip_dirs):
                return False
            
            # Include supported files
            return (file_ext in self.supported_extensions or 
                    filename.lower() in self.supported_extensions or
                    self.is_likely_code_file(filename, relative_path))
                    
        except Exception as e:
            logger.warning(f"⚠️ Error in file filtering for {relative_path}: {e}")
            return False

    def is_likely_code_file(self, filename: str, relative_path: str) -> bool:
        """Detect if file is likely code even without known extension"""
        # Common code file patterns
        code_patterns = ['makefile', 'dockerfile', 'jenkinsfile', 'vagrantfile', 'rakefile']
        if filename.lower() in code_patterns:
            return True
            
        # Files in source directories
        source_dirs = ['src', 'lib', 'app', 'components', 'utils', 'helpers', 'core', 'modules']
        if any(dir_name in relative_path.lower() for dir_name in source_dirs):
            return True
            
        return False

    def process_file_intelligently(self, file_path: str, relative_path: str) -> Optional[CodeFile]:
        """Human-level intelligent file processing with comprehensive error handling"""
        try:
            file_ext = Path(file_path).suffix.lower()
            filename = Path(file_path).name.lower()
            
            # Determine language
            language = self.supported_extensions.get(file_ext) or \
                      self.supported_extensions.get(filename, 'Unknown')
            
            # Read file content with error handling
            content = self.read_file_intelligently(file_path)
            if not content:
                logger.debug(f"⏭️ Skipping empty file: {relative_path}")
                return None
            
            # Skip files that are too large for processing
            if len(content) > 200000:  # 200KB limit for content processing
                content = content[:200000] + "\n... [File truncated due to size]"
                logger.debug(f"✂️ Truncated large file: {relative_path}")
            
            # Extract comprehensive metadata
            functions, classes, imports, todos = self.extract_comprehensive_metadata(content, language)
            
            # Calculate complexity score
            complexity_score = self.calculate_complexity_score(content, functions, classes, language)
            
            return CodeFile(
                path=relative_path,
                content=content,
                file_type=file_ext,
                language=language,
                size=len(content),
                lines=len(content.splitlines()),
                functions=functions,
                classes=classes,
                imports=imports,
                todos=todos,
                complexity_score=complexity_score
            )
            
        except Exception as e:
            logger.warning(f"⚠️ Error processing file {relative_path}: {e}")
            return None

    def read_file_intelligently(self, file_path: str) -> str:
        """Read file content with comprehensive encoding handling and format detection"""
        try:
            file_ext = Path(file_path).suffix.lower()
            
            if file_ext == '.ipynb':
                return self.read_jupyter_notebook(file_path)
            elif file_ext in ['.json']:
                return self.read_json_file(file_path)
            else:
                return self.read_text_file(file_path)
                
        except Exception as e:
            logger.warning(f"⚠️ Could not read file {file_path}: {e}")
            return ""

    def read_jupyter_notebook(self, file_path: str) -> str:
        """Read Jupyter notebook with comprehensive error handling"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                notebook = json.load(f)
            
            content_parts = []
            cell_count = 0
            
            for cell in notebook.get('cells', []):
                try:
                    cell_type = cell.get('cell_type', 'unknown')
                    source = cell.get('source', [])
                    
                    if isinstance(source, list):
                        source_text = ''.join(source)
                    else:
                        source_text = str(source)
                    
                    if source_text.strip():
                        if cell_type == 'code':
                            content_parts.append(f"# Notebook Cell {cell_count + 1} (Code)\n{source_text}\n")
                        elif cell_type == 'markdown':
                            content_parts.append(f"# Notebook Cell {cell_count + 1} (Markdown)\n{source_text}\n")
                        cell_count += 1
                        
                except Exception as cell_error:
                    logger.warning(f"⚠️ Error processing notebook cell: {cell_error}")
                    continue
            
            return '\n'.join(content_parts)
            
        except json.JSONDecodeError as e:
            logger.warning(f"⚠️ Invalid JSON in notebook {file_path}: {e}")
            return ""
        except Exception as e:
            logger.warning(f"⚠️ Error reading notebook {file_path}: {e}")
            return ""

    def read_json_file(self, file_path: str) -> str:
        """Read JSON file with formatting"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return json.dumps(data, indent=2)
        except Exception as e:
            logger.warning(f"⚠️ Error reading JSON file {file_path}: {e}")
            return self.read_text_file(file_path)

    def read_text_file(self, file_path: str) -> str:
        """Read text file with multiple encoding attempts"""
        encodings = ['utf-8', 'utf-16', 'latin-1', 'cp1252', 'ascii']
        
        for encoding in encodings:
            try:
                with open(file_path, 'r', encoding=encoding, errors='ignore') as f:
                    content = f.read()
                    # Validate that the content is readable
                    if len(content.strip()) > 0:
                        return content
            except (UnicodeDecodeError, UnicodeError):
                continue
            except Exception as e:
                logger.warning(f"⚠️ Error reading file {file_path} with encoding {encoding}: {e}")
                continue
        
        logger.warning(f"⚠️ Could not read file {file_path} with any encoding")
        return ""

    def extract_comprehensive_metadata(self, content: str, language: str) -> Tuple[List[Dict], List[Dict], List[str], List[str]]:
        """Human-level metadata extraction with comprehensive analysis"""
        functions = []
        classes = []
        imports = []
        todos = []
        
        lines = content.splitlines()
        
        # Extract TODOs, FIXMEs, and other comments (universal)
        for i, line in enumerate(lines):
            line_lower = line.lower()
            if any(keyword in line_lower for keyword in ['todo', 'fixme', 'hack', 'xxx', 'bug', 'note', 'warning']):
                todos.append({
                    'line': i + 1,
                    'content': line.strip(),
                    'type': self.classify_todo(line)
                })
        
        # Language-specific analysis
        if language == 'Python':
            functions, classes, imports = self.analyze_python_code(content, lines)
        elif language in ['JavaScript', 'TypeScript', 'JavaScript/React', 'TypeScript/React']:
            functions, classes, imports = self.analyze_javascript_code(content, lines)
        elif language == 'Java':
            functions, classes, imports = self.analyze_java_code(content, lines)
        elif language == 'Jupyter Notebook':
            # Already processed in read_file_intelligently
            functions, classes, imports = self.analyze_python_code(content, lines)
        
        return functions[:50], classes[:25], imports[:30], todos[:20]

    def analyze_python_code(self, content: str, lines: List[str]) -> Tuple[List[Dict], List[Dict], List[str]]:
        """Deep Python code analysis with AST parsing"""
        functions = []
        classes = []
        imports = []
        
        try:
            tree = ast.parse(content)
            
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    # Extract function details
                    func_info = {
                        'name': node.name,
                        'line': node.lineno,
                        'args': [arg.arg for arg in node.args.args],
                        'docstring': ast.get_docstring(node),
                        'is_async': isinstance(node, ast.AsyncFunctionDef),
                        'decorators': [self.get_decorator_name(dec) for dec in node.decorator_list]
                    }
                    functions.append(func_info)
                
                elif isinstance(node, ast.ClassDef):
                    # Extract class details
                    methods = [n.name for n in node.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
                    class_info = {
                        'name': node.name,
                        'line': node.lineno,
                        'bases': [self.get_base_name(base) for base in node.bases],
                        'docstring': ast.get_docstring(node),
                        'methods': methods,
                        'decorators': [self.get_decorator_name(dec) for dec in node.decorator_list]
                    }
                    classes.append(class_info)
                
                elif isinstance(node, (ast.Import, ast.ImportFrom)):
                    # Extract import details
                    if isinstance(node, ast.Import):
                        for alias in node.names:
                            imports.append(f"import {alias.name}")
                    else:
                        module = node.module or ""
                        for alias in node.names:
                            imports.append(f"from {module} import {alias.name}")
        
        except SyntaxError:
            # Fallback to regex parsing for invalid Python syntax
            functions, classes, imports = self.analyze_python_fallback(lines)
        
        return functions, classes, imports

    def analyze_javascript_code(self, content: str, lines: List[str]) -> Tuple[List[Dict], List[Dict], List[str]]:
        """Deep JavaScript/TypeScript analysis with pattern recognition"""
        functions = []
        classes = []
        imports = []
        
        for i, line in enumerate(lines):
            line = line.strip()
            
            # Function detection
            if 'function' in line or '=>' in line:
                if 'function ' in line:
                    match = re.search(r'function\s+(\w+)', line)
                    if match:
                        functions.append({
                            'name': match.group(1),
                            'line': i + 1,
                            'type': 'function',
                            'is_async': 'async' in line
                        })
                elif '=>' in line:
                    match = re.search(r'(\w+)\s*[=:]\s*.*=>', line)
                    if match:
                        functions.append({
                            'name': match.group(1),
                            'line': i + 1,
                            'type': 'arrow_function',
                            'is_async': 'async' in line
                        })
            
            # Class detection
            if line.startswith(('class ', 'export class ')):
                match = re.search(r'class\s+(\w+)', line)
                if match:
                    classes.append({
                        'name': match.group(1),
                        'line': i + 1,
                        'is_export': 'export' in line,
                        'extends': 'extends' in line
                    })
            
            # Import detection
            if line.startswith(('import ', 'const ', 'require')):
                imports.append(line)
        
        return functions, classes, imports

    def analyze_java_code(self, content: str, lines: List[str]) -> Tuple[List[Dict], List[Dict], List[str]]:
        """Deep Java code analysis with comprehensive pattern recognition"""
        functions = []
        classes = []
        imports = []
        
        for i, line in enumerate(lines):
            line = line.strip()
            
            # Method detection
            if ('public ' in line or 'private ' in line or 'protected ' in line) and '(' in line:
                match = re.search(r'(public|private|protected)\s+.*\s+(\w+)\s*\(', line)
                if match:
                    functions.append({
                        'name': match.group(2),
                        'line': i + 1,
                        'visibility': match.group(1),
                        'is_static': 'static' in line
                    })
            
            # Class detection
            if line.startswith(('public class ', 'class ', 'abstract class ', 'interface ')):
                match = re.search(r'(?:class|interface)\s+(\w+)', line)
                if match:
                    classes.append({
                        'name': match.group(1),
                        'line': i + 1,
                        'is_public': 'public' in line,
                        'is_abstract': 'abstract' in line,
                        'is_interface': 'interface' in line
                    })
            
            # Import detection
            if line.startswith('import '):
                imports.append(line)
        
        return functions, classes, imports

    def get_decorator_name(self, decorator) -> str:
        """Extract decorator name from AST node"""
        try:
            if isinstance(decorator, ast.Name):
                return decorator.id
            elif isinstance(decorator, ast.Attribute):
                return decorator.attr
            return str(decorator)
        except:
            return "unknown"

    def get_base_name(self, base) -> str:
        """Extract base class name from AST node"""
        try:
            if isinstance(base, ast.Name):
                return base.id
            elif isinstance(base, ast.Attribute):
                return base.attr
            return str(base)
        except:
            return "unknown"

    def analyze_python_fallback(self, lines: List[str]) -> Tuple[List[Dict], List[Dict], List[str]]:
        """Fallback Python analysis using regex when AST parsing fails"""
        functions = []
        classes = []
        imports = []
        
        for i, line in enumerate(lines):
            line = line.strip()
            
            if line.startswith(('def ', 'async def ')):
                match = re.search(r'(?:async\s+)?def\s+(\w+)', line)
                if match:
                    functions.append({
                        'name': match.group(1),
                        'line': i + 1,
                        'is_async': line.startswith('async def')
                    })
            
            elif line.startswith('class '):
                match = re.search(r'class\s+(\w+)', line)
                if match:
                    classes.append({
                        'name': match.group(1),
                        'line': i + 1
                    })
            
            elif line.startswith(('import ', 'from ')):
                imports.append(line)
        
        return functions, classes, imports

    def classify_todo(self, line: str) -> str:
        """Classify TODO type based on content"""
        line_lower = line.lower()
        if any(word in line_lower for word in ['fixme', 'bug', 'error']):
            return 'bug'
        elif 'todo' in line_lower:
            return 'enhancement'
        elif any(word in line_lower for word in ['hack', 'xxx', 'temporary']):
            return 'refactor'
        return 'general'

    def calculate_complexity_score(self, content: str, functions: List[Dict], 
                                 classes: List[Dict], language: str) -> int:
        """Calculate comprehensive code complexity score"""
        # Base complexity from lines of code
        base_score = len(content.splitlines())
        
        # Structural complexity
        function_score = len(functions) * 3
        class_score = len(classes) * 8
        
        # Cyclomatic complexity indicators
        complexity_keywords = ['if', 'for', 'while', 'try', 'catch', 'switch']
        keyword_score = sum(content.lower().count(keyword) for keyword in complexity_keywords)
        
        total_score = base_score + function_score + class_score + keyword_score
        
        # Normalize to reasonable range
        return min(max(total_score, 1), 2000)

    def store_repository_data(self, repo_url: str, repo_name: str, code_files: List[CodeFile]) -> int:
        """Store repository analysis in PostgreSQL with comprehensive error handling"""
        cursor = self.conn.cursor()
        
        try:
            # Calculate summary statistics
            total_lines = sum(f.lines for f in code_files)
            languages = {}
            for file in code_files:
                lang = file.language
                if lang not in languages:
                    languages[lang] = {'files': 0, 'lines': 0}
                languages[lang]['files'] += 1
                languages[lang]['lines'] += file.lines
            
            # Detect project type and purpose
            project_type, main_purpose = self.detect_project_type_and_purpose(code_files)
            
            # Insert repository record with conflict handling
            cursor.execute("""
                INSERT INTO repositories (repo_url, repo_name, total_files, total_lines, 
                                        languages, project_type, main_purpose)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (repo_url) DO UPDATE SET
                    analyzed_at = NOW(),
                    total_files = EXCLUDED.total_files,
                    total_lines = EXCLUDED.total_lines,
                    languages = EXCLUDED.languages,
                    project_type = EXCLUDED.project_type,
                    main_purpose = EXCLUDED.main_purpose
                RETURNING id
            """, (repo_url, repo_name, len(code_files), total_lines, 
                  Json(languages), project_type, main_purpose))
            
            repo_id = cursor.fetchone()['id']
            
            # Clean up old data for this repository
            cursor.execute("DELETE FROM code_files WHERE repo_id = %s", (repo_id,))
            cursor.execute("DELETE FROM project_insights WHERE repo_id = %s", (repo_id,))
            
            # Store file data with embeddings
            for file in code_files:
                try:
                    # Generate embedding with error handling
                    embedding_text = f"File: {file.path} Language: {file.language} Content: {file.content[:2000]}"
                    embedding = self.embedding_model.encode([embedding_text]).tolist()[0]
                    
                    cursor.execute("""
                        INSERT INTO code_files (repo_id, file_path, file_type, language, 
                                              content, lines, functions, classes, imports,
                                              complexity_score, embedding)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    """, (repo_id, file.path, file.file_type, file.language,
                          file.content, file.lines, Json(file.functions), 
                          Json(file.classes), Json(file.imports),
                          file.complexity_score, embedding))
                except Exception as embed_error:
                    logger.warning(f"⚠️ Failed to create embedding for {file.path}: {embed_error}")
                    # Store without embedding
                    cursor.execute("""
                        INSERT INTO code_files (repo_id, file_path, file_type, language, 
                                              content, lines, functions, classes, imports,
                                              complexity_score)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    """, (repo_id, file.path, file.file_type, file.language,
                          file.content, file.lines, Json(file.functions), 
                          Json(file.classes), Json(file.imports),
                          file.complexity_score))
            
            # Generate and store project insights
            insights = self.generate_human_insights(code_files, repo_url)
            for insight in insights:
                cursor.execute("""
                    INSERT INTO project_insights (repo_id, insight_type, title, 
                                                description, confidence_score, file_references)
                    VALUES (%s, %s, %s, %s, %s, %s)
                """, (repo_id, insight.insight_type, insight.title, 
                      insight.description, insight.confidence_score, 
                      Json(insight.file_references)))
            
            self.conn.commit()
            logger.info(f"✅ Stored analysis for {len(code_files)} files in database")
            return repo_id
            
        except Exception as e:
            self.conn.rollback()
            logger.error(f"❌ Database storage error: {e}")
            raise
        finally:
            cursor.close()

    def detect_project_type_and_purpose(self, code_files: List[CodeFile]) -> Tuple[str, str]:
        """Human-level project type and purpose detection with comprehensive analysis"""
        file_paths = [f.path.lower() for f in code_files]
        languages = set(f.language for f in code_files)
        
        # Detect project type
        project_type = "Unknown"
        
        if any('package.json' in path for path in file_paths):
            project_type = "JavaScript/Node.js Project"
        elif 'Python' in languages:
            if any('.ipynb' in f.path for f in code_files):
                project_type = "Machine Learning/Data Science Project"
            else:
                project_type = "Python Application"
        elif 'Java' in languages:
            project_type = "Java Application"
        
        # Detect main purpose
        main_purpose = self.analyze_project_purpose(code_files)
        
        return project_type, main_purpose

    def analyze_project_purpose(self, code_files: List[CodeFile]) -> str:
        """Analyze the main purpose of the project"""
        # Look for README files first
        readme_content = ""
        for file in code_files:
            if 'readme' in file.path.lower():
                readme_content = file.content.lower()
                break
        
        if readme_content:
            if any(keyword in readme_content for keyword in ['machine learning', 'ml', 'neural network', 'ai']):
                return "Machine Learning and AI Development"
            elif any(keyword in readme_content for keyword in ['web app', 'website', 'api']):
                return "Web Application Development"
        
        # Analyze code patterns
        all_content = ' '.join(f.content.lower() for f in code_files[:10])
        
        if any(keyword in all_content for keyword in ['tensorflow', 'pytorch', 'sklearn']):
            return "Machine Learning and Data Science"
        elif any(keyword in all_content for keyword in ['react', 'vue', 'angular']):
            return "Frontend Web Development"
        elif any(keyword in all_content for keyword in ['express', 'flask', 'django', 'fastapi']):
            return "Backend Web Development"
        
        return "General Software Development"

    def generate_human_insights(self, code_files: List[CodeFile], repo_url: str) -> List[ProjectInsight]:
        """Generate human-level insights about the project"""
        insights = []
        
        # Analyze TODOs and technical debt
        all_todos = []
        for file in code_files:
            for todo in file.todos:
                all_todos.append((file.path, todo))
        
        if all_todos:
            insights.append(ProjectInsight(
                insight_type="improvement",
                title="Development Tasks and Technical Debt",
                description=f"Found {len(all_todos)} TODO items and improvement opportunities.",
                confidence_score=0.9,
                file_references=[todo[0] for todo in all_todos[:5]]
            ))
        
        # Analyze complexity
        high_complexity_files = [f for f in code_files if f.complexity_score > 200]
        if high_complexity_files:
            insights.append(ProjectInsight(
                insight_type="quality",
                title="Code Complexity Analysis",
                description=f"Found {len(high_complexity_files)} files with high complexity that may need refactoring.",
                confidence_score=0.8,
                file_references=[f.path for f in high_complexity_files[:3]]
            ))
        
        return insights

    def search_similar_code_pgvector(self, query: str, repo_id: int, k: int = 8) -> List[Dict]:
        """Search for similar code using pgvector"""
        try:
            # Generate query embedding
            query_embedding = self.embedding_model.encode([query]).tolist()[0]
            
            cursor = self.conn.cursor()
            cursor.execute("""
                SELECT file_path, language, content, lines, functions, classes,
                       (embedding <=> %s::vector) as distance
                FROM code_files 
                WHERE repo_id = %s AND embedding IS NOT NULL
                ORDER BY embedding <=> %s::vector 
                LIMIT %s
            """, (query_embedding, repo_id, query_embedding, k))
            
            results = []
            for row in cursor.fetchall():
                similarity_score = max(0, 1 - row['distance'])
                results.append({
                    "path": row['file_path'],
                    "language": row['language'],
                    "content": row['content'],
                    "lines": row['lines'],
                    "functions": row['functions'] or [],
                    "classes": row['classes'] or [],
                    "similarity_score": similarity_score
                })
            
            cursor.close()
            return results
            
        except Exception as e:
            logger.warning(f"⚠️ Vector search failed: {e}")
            return []

    def generate_comprehensive_context_with_pgvector(self, repo_id: int, user_needs: str) -> str:
        """Generate comprehensive context using PostgreSQL + pgvector"""
        cursor = self.conn.cursor()
        
        try:
            # Get repository info
            cursor.execute("""
                SELECT repo_url, repo_name, total_files, total_lines, languages, 
                       project_type, main_purpose 
                FROM repositories WHERE id = %s
            """, (repo_id,))
            repo_info = cursor.fetchone()
            
            if not repo_info:
                raise ValueError(f"Repository with id {repo_id} not found")
            
            # Get project insights
            cursor.execute("""
                SELECT insight_type, title, description, confidence_score, file_references
                FROM project_insights WHERE repo_id = %s
                ORDER BY confidence_score DESC
            """, (repo_id,))
            insights = cursor.fetchall()
            
            # Search for relevant files
            relevant_files = self.search_similar_code_pgvector(user_needs, repo_id, k=8)
            
            context = f"""# HUMAN-LEVEL REPOSITORY ANALYSIS

## PROJECT OVERVIEW
- **Repository**: {repo_info['repo_name']}
- **Type**: {repo_info['project_type']}
- **Main Purpose**: {repo_info['main_purpose']}
- **Total Files**: {repo_info['total_files']}
- **Total Lines**: {repo_info['total_lines']:,}
- **Languages**: {', '.join(repo_info['languages'].keys())}

## USER REQUIREMENTS
{user_needs}

## INTELLIGENT INSIGHTS
"""
            
            # Add project insights
            for insight in insights:
                context += f"""
### {insight['title']} ({insight['insight_type'].title()})
**Confidence**: {insight['confidence_score']:.1%}
**Description**: {insight['description']}
**Files**: {', '.join(insight['file_references'][:3])}
"""
            
            # Add most relevant files
            context += "\n## MOST RELEVANT FILES:\n"
            for file_info in relevant_files:
                context += f"""
#### {file_info['path']} (Relevance: {file_info['similarity_score']:.1%})
- **Language**: {file_info['language']}
- **Lines**: {file_info['lines']}
- **Functions**: {len(file_info['functions'])}
- **Classes**: {len(file_info['classes'])}

**Code Preview:**
{file_info['content'][:2000]}

text """


            
            cursor.close()
            return context
            
        except Exception as e:
            if cursor:
                cursor.close()
            logger.error(f"❌ Context generation failed: {e}")
            raise

    def generate_documentation_with_human_intelligence(self, repo_url: str, user_needs: str) -> Dict[str, Any]:
        """Generate human-level documentation using PostgreSQL + pgvector"""
        try:
            logger.info(f"🚀 Starting documentation generation")
            
            # Step 1: Repository analysis
            repo_id, code_files = self.clone_and_analyze_repo(repo_url)
            
            # Step 2: Context generation
            full_context = self.generate_comprehensive_context_with_pgvector(repo_id, user_needs)
            
            # Step 3: LLM documentation generation
            documentation = self.generate_docs_with_llm_enhanced(full_context, user_needs, repo_url)
            
            # Step 4: Project analysis
            project_analysis = self.create_comprehensive_project_analysis(repo_id)
            
            result = {
                "repo_url": repo_url,
                "user_needs": user_needs,
                "documentation": documentation,
                "project_analysis": project_analysis,
                "database_stored": True,
                "total_files_analyzed": len(code_files),
                "human_level_analysis": True,
                "generation_timestamp": datetime.now().isoformat()
            }
            
            logger.info("🎉 Documentation generation completed successfully")
            return result
            
        except Exception as e:
            logger.error(f"❌ Documentation generation failed: {e}")
            raise

    def generate_docs_with_llm_enhanced(self, context: str, user_needs: str, repo_url: str) -> str:
        """Generate documentation with enhanced LLM prompting and SSL fix"""
        try:
            logger.info("🔄 Initializing Groq client with SSL fix...")
            
            # CRITICAL: Fix SSL certificate issue (this made your test work)
            import certifi
            import os
            os.environ["SSL_CERT_FILE"] = certifi.where()
            
            api_key = os.getenv("GROQ_API_KEY")
            if not api_key:
                raise ValueError("GROQ_API_KEY environment variable not set")
            
            from groq import Groq
            client = Groq(api_key=api_key)
            logger.info("✅ Groq client initialized successfully")
            
            prompt = f"""You are a senior software architect and technical writer with 15+ years of experience. 
    Analyze this codebase with deep technical insight.

    COMPREHENSIVE CODEBASE ANALYSIS:
    {context}

    USER REQUIREMENTS: {user_needs}
    REPOSITORY: {repo_url}

    Generate comprehensive documentation with these sections:

    1. **Executive Summary**: Project purpose and value
    2. **Architecture Overview**: System design and patterns
    3. **Technology Stack**: Technologies used and rationale
    4. **Key Components**: Critical files and functions
    5. **Code Quality**: Strengths and improvements
    6. **Setup Guide**: Installation and configuration
    7. **Usage Examples**: Code examples and tutorials
    8. **API Documentation**: Interface documentation
    9. **Development Workflow**: Contributing guidelines
    10. **Future Roadmap**: Improvement suggestions

    Write as if mentoring a new team member. Be specific and actionable.
    """
            
            logger.info("🔄 Calling Groq API...")
            completion = client.chat.completions.create(
                model="qwen/qwen3-32b",  # CHANGED: Use working model from your test
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
                max_completion_tokens=12000,
                top_p=0.9,
                stream=False,
            )
            
            result = completion.choices[0].message.content
            
            if not result or len(result.strip()) < 500:
                raise ValueError("Generated documentation is too short")
            
            logger.info(f"✅ Documentation generated: {len(result)} characters")
            return result
            
        except Exception as e:
            logger.error(f"❌ LLM generation failed: {e}")
            logger.error(f"Error type: {type(e).__name__}")
            return self.generate_fallback_documentation(context, user_needs, repo_url)

    def generate_fallback_documentation(self, context: str, user_needs: str, repo_url: str) -> str:
        """Generate fallback documentation when LLM fails"""
        logger.info("🔄 Generating fallback documentation")
        
        return f"""# Repository Documentation

**Repository**: {repo_url}
**User Requirements**: {user_needs}
**Generated**: {datetime.now().isoformat()}

## Executive Summary
This documentation was generated automatically for the repository analysis.

## Context Information
{context[:2000]}...

## Note
This is a fallback documentation generated when the AI service was unavailable.
For comprehensive AI-powered documentation, please try again later.
"""

    def create_comprehensive_project_analysis(self, repo_id: int) -> Dict[str, Any]:
        """Create comprehensive project analysis from database"""
        cursor = self.conn.cursor()
        
        try:
            # Get repository data
            cursor.execute("SELECT * FROM repositories WHERE id = %s", (repo_id,))
            repo_data = cursor.fetchone()
            
            # Get file statistics
            cursor.execute("""
                SELECT language, COUNT(*) as file_count, 
                       SUM(lines) as total_lines,
                       AVG(complexity_score) as avg_complexity
                FROM code_files WHERE repo_id = %s 
                GROUP BY language
                ORDER BY total_lines DESC
            """, (repo_id,))
            language_stats = cursor.fetchall()
            
            # Get insights
            cursor.execute("""
                SELECT insight_type, COUNT(*) as count
                FROM project_insights WHERE repo_id = %s
                GROUP BY insight_type
            """, (repo_id,))
            insight_summary = cursor.fetchall()
            
            return {
                "repository_info": dict(repo_data),
                "language_distribution": [dict(row) for row in language_stats],
                "insight_summary": [dict(row) for row in insight_summary],
                "analysis_quality": "human_level"
            }
            
        finally:
            cursor.close()

    def cleanup_resources(self):
        """Clean up all resources"""
        try:
            # Close database connection
            if hasattr(self, 'conn') and self.conn:
                self.conn.close()
                logger.info("✅ Database connection closed")
            
            # Clean up temporary directories
            for temp_dir in self.temp_directories:
                if os.path.exists(temp_dir):
                    self.safe_remove_tree(temp_dir)
            
            self.temp_directories.clear()
            
        except Exception as e:
            logger.warning(f"⚠️ Error during cleanup: {e}")

    def __del__(self):
        """Destructor with proper cleanup"""
        self.cleanup_resources()

# Main function for API
def generate_human_level_repo_docs(repo_url: str, user_needs: str) -> Dict[str, Any]:
    """Generate human-level repository documentation using PostgreSQL + pgvector"""
    analyzer = None
    try:
        logger.info(f"🎯 Starting repository documentation generation")
        analyzer = HumanLevelRepoAnalyzer()
        result = analyzer.generate_documentation_with_human_intelligence(repo_url, user_needs)
        return result
    finally:
        if analyzer:
            analyzer.cleanup_resources()
