#pip install ollama

import ollama
response = ollama.chat(model='gemma:2b-instruct', messages=[
  {
    'role': 'user',
    'content': 'Why is the sky blue?',
  },
])
print(response['message']['content'])

#--------------------------------------------------

resp = ollama.generate(model='llama2', prompt='Why is the sky blue?')
print(resp)

#--------------------------------------------------

emb = ollama.embeddings(model='gemma:2b-instruct', prompt='They sky is blue because of rayleigh scattering')
print(emb)

#--------------------------------------------------

stream = ollama.chat(
    model='gemma:2b-instruct',
    messages=[{'role': 'user', 'content': 'Why is the sky blue?'}],
    stream=True,
)

for chunk in stream:
  print(chunk['message']['content'], end='', flush=True)

#--------------------------------------------------