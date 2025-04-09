import gradio as gr
from dotenv import load_dotenv
from openai import OpenAI
import json
import faiss
import numpy as np

load_dotenv()

client = OpenAI()

chunks = []
with open("resources/chunks.json", "r") as f:
    chunks = json.load(f)

index = faiss.read_index("resources/docbot.index")

def search_index(query, k=5):
    query_embedding = client.embeddings.create(
        model="text-embedding-3-small",
        input=query
    ).data[0].embedding

    query_vector = np.array([query_embedding], dtype="float32")

    _, I = index.search(query_vector, k)

    return [chunks[i] for i in I[0]]

def query(message, history):
    top_k_chunks = search_index(message)

    context = [json.dumps(chunk["raw_yaml"], indent=4) for chunk in top_k_chunks]
    context_str = "\n\n".join(context)

    sources = list({chunk["source"] for chunk in top_k_chunks if chunk["source"] != ""})
    sources_str = "\n\nRead More:\n\n" + "\n\n".join(sources)

    response = client.chat.completions.create(
        model="gpt-4o",
        temperature=0,
        messages=[
            {
                "role": "developer",
                "content": (
                    "You are an expert developer assistant who helps users understand how to use UPS APIs. "
                    "Your answers are based solely on internal documentation provided to you. "
                    "If the answer is not in the context, say you don't know. "
                    "Respond clearly and concisely, using examples when relevant."
                )
            },
            {
                "role": "system",
                "content": f"Context from internal documentation:\n\n{context_str}"
            },
            {
                "role": "user",
                "content": message
            }
        ],
        stream=True
    )

    full_response = ""
    
    for chunk in response:
        delta = chunk.choices[0].delta.content
        if delta:
            full_response += delta
            yield full_response

    if sources:
        yield full_response + sources_str

    

demo = gr.ChatInterface(
    fn=query,
    type="messages"
)

demo.launch()