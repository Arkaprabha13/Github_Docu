import os
import certifi
from dotenv import load_dotenv

# Fix SSL certificate issue first
os.environ["SSL_CERT_FILE"] = certifi.where()

def test_groq():
    """Simple Groq API test"""
    
    # Load environment variables
    load_dotenv()
    
    # Check API key
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        print("❌ GROQ_API_KEY not found in .env file")
        return False
    
    print(f"🔑 API Key found: {api_key[:10]}...")
    
    try:
        # Import and initialize Groq
        from groq import Groq
        client = Groq(api_key=api_key)
        print("✅ Groq client initialized successfully")
        
        # Test simple completion
        print("🔄 Testing API call...")
        
        completion = client.chat.completions.create(
            model="llama3-8b-8192",  # Fast model for testing
            messages=[
                {"role": "user", "content": "Say 'Hello, Groq is working!' in exactly those words."}
            ],
            max_tokens=20,
            temperature=0
        )
        
        response = completion.choices[0].message.content
        print(f"✅ Groq API Response: {response}")
        
        # Test with your preferred model
        print("🔄 Testing with deepseek model...")
        
        completion2 = client.chat.completions.create(
            model="deepseek-r1-distill-llama-70b",
            messages=[
                {"role": "user", "content": "Respond with just: 'DeepSeek model working!'"}
            ],
            max_tokens=10,
            temperature=0
        )
        
        response2 = completion2.choices[0].message.content
        print(f"✅ DeepSeek Response: {response2}")
        
        print("🎉 All tests passed! Groq is working correctly.")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

if __name__ == "__main__":
    print("🧪 Simple Groq API Test")
    print("=" * 30)
    test_groq()
