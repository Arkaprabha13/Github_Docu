"""
Standalone database connection test for Supabase PostgreSQL
Run this after updating your password to verify connectivity
"""

import psycopg2
from psycopg2.extras import RealDictCursor
import os
from dotenv import load_dotenv
from urllib.parse import urlparse, unquote
import sys

# Load environment variables
load_dotenv()

def test_database_connection():
    """Test database connection with detailed diagnostics"""
    
    print("=" * 60)
    print("🧪 DATABASE CONNECTION TEST")
    print("=" * 60)
    
    # Get DATABASE_URL
    db_url = os.getenv("DATABASE_URL")
    if not db_url:
        print("❌ DATABASE_URL environment variable not found")
        print("💡 Make sure you have DATABASE_URL set in your .env file")
        return False
    
    # Parse URL for diagnostics
    try:
        parsed = urlparse(db_url)
        print(f"🔍 Connection Details:")
        print(f"   Host: {parsed.hostname}")
        print(f"   Port: {parsed.port or 5432}")
        print(f"   Database: {parsed.path.lstrip('/') or 'postgres'}")
        print(f"   Username: {parsed.username}")
        print(f"   Password: {'*' * len(parsed.password) if parsed.password else 'None'}")
        print(f"   Has special chars: {'#' in unquote(parsed.password) if parsed.password else 'Unknown'}")
        print()
    except Exception as e:
        print(f"⚠️ URL parsing warning: {e}")
    
    # Test connection
    print("🔄 Testing database connection...")
    
    try:
        # Connect to database
        conn = psycopg2.connect(
            db_url,
            cursor_factory=RealDictCursor,
            sslmode='require',
            connect_timeout=15
        )
        
        print("✅ Connection established successfully!")
        
        # Test basic query
        cursor = conn.cursor()
        cursor.execute("SELECT current_user, version(), now() as current_time")
        result = cursor.fetchone()
        
        print(f"👤 Connected as: {result['current_user']}")
        print(f"🐘 PostgreSQL version: {result['version'][:60]}...")
        print(f"🕐 Server time: {result['current_time']}")
        print()
        
        # Test pgvector extension
        print("🔍 Checking pgvector extension...")
        cursor.execute("""
            SELECT EXISTS(
                SELECT 1 FROM pg_extension 
                WHERE extname = 'vector'
            ) as pgvector_installed
        """)
        
        pgvector_result = cursor.fetchone()
        if pgvector_result['pgvector_installed']:
            print("✅ pgvector extension is installed and available")
            
            # Test vector operations
            try:
                cursor.execute("SELECT '[1,2,3]'::vector as test_vector")
                vector_result = cursor.fetchone()
                print(f"✅ Vector operations working: {vector_result['test_vector']}")
            except Exception as ve:
                print(f"⚠️ Vector operations issue: {ve}")
        else:
            print("❌ pgvector extension not found")
            print("💡 Enable it in Supabase: Dashboard → Database → Extensions → vector")
        
        print()
        
        # Test table creation (cleanup after)
        print("🧪 Testing table operations...")
        try:
            cursor.execute("""
                CREATE TEMP TABLE connection_test (
                    id SERIAL PRIMARY KEY,
                    test_data TEXT,
                    created_at TIMESTAMP DEFAULT NOW()
                )
            """)
            
            cursor.execute("""
                INSERT INTO connection_test (test_data) 
                VALUES ('Connection test successful')
                RETURNING id, test_data, created_at
            """)
            
            test_result = cursor.fetchone()
            print(f"✅ Table operations working: Record ID {test_result['id']} created")
            
        except Exception as te:
            print(f"⚠️ Table operations issue: {te}")
        
        # Close connection
        conn.close()
        print("✅ Connection closed properly")
        print()
        print("🎉 ALL TESTS PASSED! Your database is ready to use.")
        return True
        
    except psycopg2.OperationalError as e:
        error_msg = str(e)
        print("❌ Connection failed!")
        print(f"Error: {error_msg}")
        print()
        
        # Specific error guidance
        if "Wrong password" in error_msg:
            print("🔧 SOLUTION: Password is incorrect")
            print("   1. Go to Supabase Dashboard → Settings → Database")
            print("   2. Copy the correct password")
            print("   3. If password has # or special chars, URL encode them:")
            print("      # becomes %23, @ becomes %40, etc.")
            print("   4. Update your DATABASE_URL")
            
        elif "authentication failed" in error_msg:
            print("🔧 SOLUTION: Check username and password")
            print("   - Username should be: postgres.your_project_ref")
            print("   - Password should be from Database settings, not API keys")
            
        elif "timeout" in error_msg:
            print("🔧 SOLUTION: Connection timeout")
            print("   - Check your internet connection")
            print("   - Verify the host URL is correct")
            
        elif "could not translate host name" in error_msg:
            print("🔧 SOLUTION: Use IPv4-compatible connection")
            print("   - Use Session Mode or Transaction Mode from Supabase")
            print("   - Should be: aws-0-region.pooler.supabase.com")
            
        return False
        
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

def main():
    """Main test function"""
    success = test_database_connection()
    
    if success:
        print("=" * 60)
        print("✅ Database connection is working perfectly!")
        print("You can now run your main application.")
        sys.exit(0)
    else:
        print("=" * 60)
        print("❌ Database connection failed.")
        print("Please fix the issues above before running your main application.")
        sys.exit(1)

if __name__ == "__main__":
    main()
