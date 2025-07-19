# test_ssl.py
import ssl
import certifi

print("Python's default certificate bundle:", certifi.where())
print("SSL default context:", ssl.create_default_context())

# Test with explicit certificate path
import os
os.environ['SSL_CERT_FILE'] = certifi.where()

from groq import Groq
client = Groq(api_key=os.getenv("GROQ_API_KEY"))
print("✅ Groq client initialized successfully!")
