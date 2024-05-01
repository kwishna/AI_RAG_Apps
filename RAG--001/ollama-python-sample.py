# pip install ollama

import ollama

response = ollama.chat(model='phi3:latest', messages=[
    {
        'role': 'user',
        'content': 'Why is the sky blue?',
    },
])
print(response['message']['content'])

# --------------------------------------------------

resp = ollama.generate(model='phi3:latest', prompt='Why is the sky blue?')
print(resp)

# --------------------------------------------------

emb = ollama.embeddings(model='phi3:latest', prompt='They sky is blue because of rayleigh scattering')
print(emb)

# --------------------------------------------------

stream = ollama.chat(
    model='phi3:latest',
    messages=[{'role': 'user', 'content': 'Why is the sky blue?'}],
    stream=True,
)

for chunk in stream:
    print(chunk['message']['content'], end='', flush=True)

# --------------------------------------------------
