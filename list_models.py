import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI(base_url=os.getenv("OPENAI_BASE_URL"))
for m in sorted(client.models.list().data, key=lambda m: m.id):
    print(m.id)