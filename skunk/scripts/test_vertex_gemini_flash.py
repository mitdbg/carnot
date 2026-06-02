import sys

PROJECT_ID = 'mit-grc-free-tier'
LOCATION = 'global'
MODEL = 'vertex_ai/gemini-3.5-flash'
PROMPT = 'Tell me one stupid joke in one or two sentences.'

from google import genai
from google.genai import types

client = genai.Client(vertexai=True, project=PROJECT_ID, location=LOCATION)

print(f"Project: {PROJECT_ID}")
print(f"Location: {LOCATION}")
print(f"Model: {MODEL}")
print(f"Expected Vertex path: /projects/{PROJECT_ID}/locations/{LOCATION}/publishers/google/models/{MODEL}:generateContent")
print(f"Prompt: {PROMPT}")
print()

try:
    response = client.models.generate_content(
        model=MODEL,
        contents=PROMPT,
        config=types.GenerateContentConfig(
            temperature=0,
        ),
    )
    if hasattr(response, 'text') and response.text:
        print("Response text:")
        print(response.text)
    else:
        print("Raw response:")
        print(response)
except Exception as e:
    print(f"Error: {type(e).__name__}: {e}", file=sys.stderr)
    raise
