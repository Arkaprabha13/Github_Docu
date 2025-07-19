import os
import shutil
import tempfile
from git import Repo as GitRepo
from groq import Groq
from dotenv import load_dotenv
from typing import Dict, List, Any, Tuple
import json
from pathlib import Path
import hashlib
from dataclasses import dataclass
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from datetime import datetime
import stat
import pickle

load_dotenv()

@dataclass
class CodeFile:
    path: str
    content: str
    file_type: str
    language: str
    size: int
    lines: int
    functions: List[str]
    classes: List[str]
    imports: List[str]

class FAISSCodebaseAnalyzer:
    def __init__(self):
        self.supported_extensions = {
            # Web Technologies
            '.js': 'JavaScript', '.jsx': 'JavaScript/React', '.ts': 'TypeScript', 
            '.tsx': 'TypeScript/React', '.html': 'HTML', '.css': 'CSS', 
            '.scss': 'SASS', '.less': 'LESS', '.vue': 'Vue.js',
            
            # Backend Languages
            '.py': 'Python', '.java': 'Java', '.cpp': 'C++', '.c': 'C',
            '.cs': 'C#', '.go': 'Go', '.rs': 'Rust', '.php': 'PHP',
            '.rb': 'Ruby', '.swift': 'Swift', '.kt': 'Kotlin',
            
            # Data & Config
            '.json': 'JSON', '.yaml': 'YAML', '.yml': 'YAML', '.xml': 'XML',
            '.toml': 'TOML', '.ini': 'INI', '.env': 'Environment',
            
            # Notebooks & Scripts
            '.ipynb': 'Jupyter Notebook', '.bat': 'Batch Script', 
            '.sh': 'Shell Script', '.ps1': 'PowerShell',
            
            # Documentation
            '.md': 'Markdown', '.rst': 'reStructuredText', '.txt': 'Text',
            
            # Build & Package
            '.dockerfile': 'Docker', '.makefile': 'Makefile', '.gradle': 'Gradle',
            'package.json': 'NPM Package', 'requirements.txt': 'Python Requirements',
            'pom.xml': 'Maven', 'cargo.toml': 'Rust Package'
        }
        
        # Initialize embedding model for FAISS
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.embedding_dimension = 384  # Dimension for all-MiniLM-L6-v2
        
    def clone_and_analyze_repo(self, repo_url: str) -> Tuple[str, List[CodeFile]]:
        """Clone repository and perform comprehensive analysis"""
        temp_dir = tempfile.mkdtemp()
        print(f"🔄 Cloning repository: {repo_url}")
        
        try:
            GitRepo.clone_from(repo_url, temp_dir)
            print(f"✅ Repository cloned to: {temp_dir}")
            
            # Analyze all files
            code_files = self.analyze_all_files(temp_dir)
            print(f"📊 Analyzed {len(code_files)} files across {len(set(f.language for f in code_files))} languages")
            
            return temp_dir, code_files
        except Exception as e:
            print(f"❌ Error cloning repository: {e}")
            raise

    def analyze_all_files(self, repo_path: str) -> List[CodeFile]:
        """Comprehensive analysis of all files in repository"""
        code_files = []
        
        for root, dirs, files in os.walk(repo_path):
            # Skip version control and node_modules
            dirs[:] = [d for d in dirs if not d.startswith(('.git', 'node_modules', '__pycache__', '.pytest_cache'))]
            
            for file in files:
                file_path = os.path.join(root, file)
                relative_path = os.path.relpath(file_path, repo_path)
                
                # Get file extension and type
                file_ext = Path(file).suffix.lower()
                if file_ext not in self.supported_extensions and file not in self.supported_extensions:
                    continue
                
                try:
                    code_file = self.process_file(file_path, relative_path)
                    if code_file:
                        code_files.append(code_file)
                except Exception as e:
                    print(f"⚠️  Error processing {relative_path}: {e}")
                    continue
        
        return code_files

    def process_file(self, file_path: str, relative_path: str) -> CodeFile:
        """Process individual file and extract metadata"""
        file_ext = Path(file_path).suffix.lower()
        file_name = Path(file_path).name.lower()
        
        # Determine language
        language = self.supported_extensions.get(file_ext) or self.supported_extensions.get(file_name, 'Unknown')
        
        # Read file content
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
        except Exception:
            return None
        
        # Extract metadata based on file type
        functions, classes, imports = self.extract_code_metadata(content, language)
        
        return CodeFile(
            path=relative_path,
            content=content,
            file_type=file_ext,
            language=language,
            size=len(content),
            lines=len(content.splitlines()),
            functions=functions,
            classes=classes,
            imports=imports
        )

    def extract_code_metadata(self, content: str, language: str) -> Tuple[List[str], List[str], List[str]]:
        """Extract functions, classes, and imports from code"""
        functions = []
        classes = []
        imports = []
        
        lines = content.splitlines()
        
        if language == 'Python':
            for line in lines:
                line = line.strip()
                if line.startswith('def '):
                    func_name = line.split('(')[0].replace('def ', '').strip()
                    functions.append(func_name)
                elif line.startswith('class '):
                    class_name = line.split('(')[0].split(':')[0].replace('class ', '').strip()
                    classes.append(class_name)
                elif line.startswith(('import ', 'from ')):
                    imports.append(line)
        
        elif language in ['JavaScript', 'TypeScript', 'JavaScript/React', 'TypeScript/React']:
            for line in lines:
                line = line.strip()
                if 'function ' in line or '=>' in line:
                    if 'function ' in line:
                        func_name = line.split('function ')[1].split('(')[0].strip()
                        functions.append(func_name)
                elif line.startswith(('class ', 'export class ')):
                    class_name = line.split('{')[0].replace('class ', '').replace('export ', '').strip()
                    classes.append(class_name)
                elif line.startswith(('import ', 'const ', 'require(')):
                    imports.append(line)
        
        elif language in ['Java', 'C#']:
            for line in lines:
                line = line.strip()
                if ('public ' in line or 'private ' in line) and '(' in line and ')' in line:
                    functions.append(line.split('(')[0].split()[-1])
                elif line.startswith(('public class ', 'class ')):
                    class_name = line.split('{')[0].split()[-1]
                    classes.append(class_name)
                elif line.startswith(('import ', 'using ')):
                    imports.append(line)
        
        return functions[:20], classes[:10], imports[:15]  # Limit for performance

    def create_faiss_index(self, code_files: List[CodeFile], repo_url: str) -> Tuple[faiss.IndexFlatL2, Dict, str]:
        """Create FAISS index for RAG-style retrieval"""
        print(f"🔧 Creating FAISS index for {len(code_files)} files...")
        
        # Prepare documents for embedding
        documents = []
        metadata = {}
        
        for i, file in enumerate(code_files):
            # Create searchable document combining code and metadata
            doc_content = f"""
            File: {file.path}
            Language: {file.language}
            Functions: {', '.join(file.functions)}
            Classes: {', '.join(file.classes)}
            Imports: {', '.join(file.imports)}
            
            Code Content:
            {file.content[:2000]}...
            """
            
            documents.append(doc_content)
            metadata[i] = {
                "path": file.path,
                "language": file.language,
                "lines": file.lines,
                "functions": len(file.functions),
                "classes": len(file.classes),
                "content": file.content
            }
        
        # Generate embeddings
        print("🔄 Generating embeddings...")
        embeddings = self.embedding_model.encode(documents, convert_to_numpy=True)
        
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(embeddings)
        
        # Create FAISS index
        index = faiss.IndexFlatL2(self.embedding_dimension)
        index.add(embeddings.astype('float32'))
        
        # Generate collection name for reference
        collection_name = f"repo_{hashlib.md5(repo_url.encode()).hexdigest()[:8]}"
        
        print(f"✅ FAISS index created with {index.ntotal} documents")
        return index, metadata, collection_name

    def search_similar_code(self, query: str, index: faiss.IndexFlatL2, metadata: Dict, k: int = 5) -> List[Dict]:
        """Search for similar code using FAISS"""
        # Embed the query
        query_embedding = self.embedding_model.encode([query], convert_to_numpy=True)
        faiss.normalize_L2(query_embedding)
        
        # Search in FAISS index
        distances, indices = index.search(query_embedding.astype('float32'), k)
        
        results = []
        for i, (distance, idx) in enumerate(zip(distances[0], indices[0])):
            if idx != -1:  # Valid result
                results.append({
                    "rank": i + 1,
                    "similarity_score": 1 - distance,  # Convert distance to similarity
                    "metadata": metadata[idx]
                })
        
        return results

    def generate_comprehensive_context_with_retrieval(self, code_files: List[CodeFile], user_needs: str, 
                                                    index: faiss.IndexFlatL2, metadata: Dict) -> str:
        """Generate comprehensive context using FAISS retrieval - FIXED VERSION"""
        
        # Project overview
        languages = set(f.language for f in code_files)
        total_lines = sum(f.lines for f in code_files)
        
        context = f"""
# COMPREHENSIVE REPOSITORY ANALYSIS WITH INTELLIGENT RETRIEVAL

## PROJECT OVERVIEW
- Total Files: {len(code_files)}
- Languages/Technologies: {', '.join(sorted(languages))}
- Total Lines of Code: {total_lines}

## USER REQUIREMENTS CONTEXT
{user_needs}

## INTELLIGENT CODE RETRIEVAL RESULTS
"""
        
        # Use FAISS to retrieve most relevant files based on user needs
        relevant_results = self.search_similar_code(user_needs, index, metadata, k=8)
        
        context += "\n### MOST RELEVANT FILES FOR YOUR REQUIREMENTS:\n"
        for result in relevant_results:
            file_meta = result["metadata"]
            similarity = result["similarity_score"]
            
            context += f"\n#### {file_meta['path']} (Relevance: {similarity:.2%})\n"
            context += f"- Language: {file_meta['language']}\n"
            context += f"- Lines: {file_meta['lines']}\n"
            context += f"- Functions: {file_meta['functions']}, Classes: {file_meta['classes']}\n"
            
            # Add significant code content for high relevance files - FIXED
            if similarity > 0.3:  # High relevance threshold
                context += f"\n``` {file_meta['language']}\n"
                context += f"{file_meta['content'][:1500]}\n```\n"
        
        # Group remaining files by language
        context += "\n## COMPLETE ARCHITECTURE ANALYSIS\n"
        files_by_lang = {}
        for file in code_files:
            if file.language not in files_by_lang:
                files_by_lang[file.language] = []
            files_by_lang[file.language].append(file)
        
        # Add overview for each language
        for language, lang_files in files_by_lang.items():
            context += f"\n### {language.upper()} FILES ({len(lang_files)} files)\n"
            
            for file in lang_files[:3]:  # Limit to avoid token overflow
                context += f"\n#### {file.path}\n"
                context += f"- Lines: {file.lines}\n"
                
                if file.functions:
                    context += f"- Functions: {', '.join(file.functions[:8])}\n"
                if file.classes:
                    context += f"- Classes: {', '.join(file.classes[:5])}\n"
        
        return context

    def generate_documentation_with_full_context(self, repo_url: str, user_needs: str) -> Dict[str, Any]:
        """Main method to generate comprehensive documentation using FAISS"""
        repo_path = None
        
        try:
            # Step 1: Clone and analyze repository
            repo_path, code_files = self.clone_and_analyze_repo(repo_url)
            
            # Step 2: Create FAISS index for RAG
            faiss_index, metadata, collection_name = self.create_faiss_index(code_files, repo_url)
            
            # Step 3: Generate comprehensive context with intelligent retrieval
            full_context = self.generate_comprehensive_context_with_retrieval(
                code_files, user_needs, faiss_index, metadata
            )
            
            # Step 4: Generate documentation with LLM
            documentation = self.generate_docs_with_llm(full_context, user_needs, repo_url)
            
            # Step 5: Create project analysis
            project_analysis = self.create_project_analysis(code_files)
            
            # Step 6: Save FAISS index for future use (optional)
            index_path = self.save_faiss_index(faiss_index, metadata, collection_name)
            
            return {
                "repo_url": repo_url,
                "user_needs": user_needs,
                "documentation": documentation,
                "project_analysis": project_analysis,
                "faiss_index_path": index_path,
                "total_files_analyzed": len(code_files),
                "intelligent_retrieval_enabled": True
            }
        
        finally:
            if repo_path and os.path.exists(repo_path):
                try:
                    self.safe_remove_tree(repo_path)
                except:
                    pass

    def save_faiss_index(self, index: faiss.IndexFlatL2, metadata: Dict, collection_name: str) -> str:
        """Save FAISS index and metadata for future use"""
        index_dir = tempfile.mkdtemp()
        index_path = os.path.join(index_dir, f"{collection_name}.index")
        metadata_path = os.path.join(index_dir, f"{collection_name}_metadata.pkl")
        
        # Save FAISS index
        faiss.write_index(index, index_path)
        
        # Save metadata
        with open(metadata_path, 'wb') as f:
            pickle.dump(metadata, f)
        
        print(f"💾 FAISS index saved to: {index_dir}")
        return index_dir

    def generate_docs_with_llm(self, context: str, user_needs: str, repo_url: str) -> str:
        """Generate documentation using full codebase context and intelligent retrieval"""
        client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        
        prompt = f"""
        You are a senior software architect and technical writer analyzing a complete codebase with intelligent retrieval capabilities.

        INTELLIGENT CODEBASE ANALYSIS:
        {context}

        USER REQUIREMENTS: {user_needs}
        REPOSITORY: {repo_url}

        Generate comprehensive, professional documentation that leverages the intelligent retrieval results:
        
        1. **Executive Summary**: High-level project purpose and key insights
        2. **Architecture Overview**: System design based on retrieved relevant files
        3. **Technology Stack**: Languages, frameworks, and tools with their purposes
        4. **Key Components**: Focus on the most relevant files identified by retrieval
        5. **Code Structure**: Detailed analysis of important classes and functions
        6. **Setup & Installation**: Step-by-step setup guide
        7. **Usage Examples**: Practical code examples from retrieved files
        8. **API Documentation**: Document public interfaces found in relevant files
        9. **Development Guide**: How to extend and contribute to the project
        10. **Best Practices**: Recommendations based on code analysis

        Focus heavily on the most relevant files identified by similarity search. Use proper Markdown formatting.
        Make it comprehensive, actionable, and tailored to the user's specific needs.
        """
        
        completion = client.chat.completions.create(
            model="deepseek-r1-distill-llama-70b",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_completion_tokens=8192,
            top_p=0.9,
            stream=False,
        )
        
        return completion.choices[0].message.content

    def safe_remove_tree(self, path: str):
        """Safely remove directory tree, handling Windows permission issues"""
        try:
            shutil.rmtree(path)
        except PermissionError:
            # Handle Windows permission issues with .git files
            for root, dirs, files in os.walk(path):
                for d in dirs:
                    os.chmod(os.path.join(root, d), stat.S_IRWXU)
                for f in files:
                    os.chmod(os.path.join(root, f), stat.S_IRWXU)
            shutil.rmtree(path)

    def create_project_analysis(self, code_files: List[CodeFile]) -> Dict[str, Any]:
        """Create detailed project analysis"""
        analysis = {
            "file_breakdown": {},
            "language_distribution": {},
            "complexity_metrics": {},
            "architecture_patterns": [],
            "key_files": []
        }
        
        # Language distribution
        for file in code_files:
            lang = file.language
            if lang not in analysis["language_distribution"]:
                analysis["language_distribution"][lang] = {"files": 0, "lines": 0}
            analysis["language_distribution"][lang]["files"] += 1
            analysis["language_distribution"][lang]["lines"] += file.lines
        
        # Identify key files (high complexity or importance)
        key_files = sorted(code_files, key=lambda x: x.lines + len(x.functions) + len(x.classes), reverse=True)[:10]
        analysis["key_files"] = [{"path": f.path, "language": f.language, "lines": f.lines} for f in key_files]
        
        return analysis

# Usage function
def generate_comprehensive_repo_docs_faiss(repo_url: str, user_needs: str) -> Dict[str, Any]:
    """Generate comprehensive documentation using FAISS-powered analysis"""
    analyzer = FAISSCodebaseAnalyzer()
    return analyzer.generate_documentation_with_full_context(repo_url, user_needs)
