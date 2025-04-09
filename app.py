import gradio as gr
from dotenv import load_dotenv
from openai import OpenAI, RateLimitError
import json
import yaml
import faiss
import numpy as np
import tiktoken
from time import sleep

load_dotenv()

client = OpenAI()

encoding = tiktoken.encoding_for_model("gpt-4o")

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

    context = [yaml.dump(chunk["raw_yaml"]) for chunk in top_k_chunks]
    context_str = "\n\n".join(context)

    sources = list({chunk["source"] for chunk in top_k_chunks if chunk["source"] != ""})
    sources_str = "\n\nRead More:\n\n" + "\n\n".join(sources)

    messages = [
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
    ]
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            temperature=0,
            messages=messages,
            stream=True
        )

        partial_response = ""
    
        for chunk in response:
            delta = chunk.choices[0].delta.content
            if delta:
                partial_response += delta
                yield partial_response

        if sources:
            for token in encoding.encode(sources_str):
                partial_response += encoding.decode([token])
                yield partial_response
                sleep(0.05)

    except RateLimitError as e:
        print(f"RateLimitError: {e}")
        print(f"Query: '{message}'")
        print(f"Input Tokens: {sum([4 + len(encoding.encode(message['content'])) for message in messages])}")

        response = "Your query was unable to be completed due to an OpenAI rate limitation. Please try again later."

        partial_response = ""

        for token in encoding.encode(response):
            partial_response += encoding.decode([token])
            yield partial_response
            sleep(0.05)
            

    

demo = gr.ChatInterface(
    fn=query,
    type="messages"
)

demo.launch()