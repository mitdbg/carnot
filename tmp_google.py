import os

from google import genai


MODEL = "gemini-3.5-flash"
PROMPT = "Say hello in one short sentence."

client = genai.Client(
    vertexai=True,
    project="mit-grc-free-tier",
)

response = client.models.generate_content(
    model=MODEL,
    contents=PROMPT,
)

# print(response.text)

from skunk.retrieval.nodes.llm_wrapper import LLMWrapper
llm_wrapper = LLMWrapper()
response = llm_wrapper.call_llm(PROMPT, model='vertex_ai/gemini-3.5-flash')
print(response)

docs = [
    "Lorem ipsum dolor sit amet, consectetur adipiscing elit.",
    "Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua.",
    "Mammt nisi ut aliquip ex ea commodo consequat.",
]

emb1 = client.models.embed_content(
    model="gemini-embedding-001",
    contents=docs,
)

emb2 = client.models.embed_content(
    model="gemini-embedding-2",
    contents=docs,
)

x2 = llm_wrapper.embed_batch(model="gemini-embedding-2", batch_texts=docs)