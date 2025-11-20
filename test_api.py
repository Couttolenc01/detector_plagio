from openai import OpenAI
from dotenv import load_dotenv

# Cargar variables del archivo .env
load_dotenv()

client = OpenAI()

resp = client.chat.completions.create(
    model="gpt-4.1-mini",
    messages=[
        {"role": "user", "content": "Escribe una frase diciendo que la API funciona."}
    ],
)

print(resp.choices[0].message.content)
