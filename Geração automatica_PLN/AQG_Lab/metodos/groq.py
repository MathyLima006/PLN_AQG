
import os
import json
from pathlib import Path

from dotenv import load_dotenv
from groq import Groq

load_dotenv(
    Path(__file__).resolve().parent.parent / ".env"
)

MODELO = "openai/gpt-oss-20b"


def gerar_perguntas_groq(
    texto,
    quantidade,
    dificuldade="Médio",
    formato="Discursivas",
    modelo=MODELO
):
    chave = os.getenv("GROQ_API_KEY")

    if not chave:
        raise ValueError(
            "Configure GROQ_API_KEY no arquivo .env."
        )

    cliente = Groq(api_key=chave, timeout=45.0)

    instrucoes = """
Você é especialista em geração automática de perguntas.

Gere perguntas em português brasileiro.
Use exclusivamente as informações fornecidas.
Não invente fatos ou respostas.
Evite perguntas repetidas.
Se o texto não permitir a quantidade solicitada,
retorne menos perguntas.

Responda somente com um objeto JSON no formato:
{
  "perguntas": [
    {
      "pergunta": "Texto da pergunta",
      "resposta": "Resposta correta"
    }
  ]
}

Para múltipla escolha, coloque quatro alternativas
na pergunta e indique a alternativa correta na resposta.
"""

    prompt = f"""
Quantidade máxima: {quantidade}
Dificuldade: {dificuldade}
Formato: {formato}

Texto:
{texto}
"""

    resposta = cliente.chat.completions.create(
        model=modelo,
        messages=[
            {"role": "system", "content": instrucoes},
            {"role": "user", "content": prompt}
        ],
        response_format={"type": "json_object"},
        temperature=0.3,
        max_completion_tokens=1800
    )

    conteudo = resposta.choices[0].message.content

    if not conteudo:
        raise ValueError("O modelo retornou uma resposta vazia.")

    dados = json.loads(conteudo)

    perguntas = []

    for item in dados.get("perguntas", []):
        if not isinstance(item, dict):
            continue

        pergunta = str(item.get("pergunta", "")).strip()
        resposta = str(item.get("resposta", "")).strip()

        if pergunta and resposta:
            perguntas.append({
                "pergunta": pergunta,
                "resposta": resposta,
                "regra": "LLM via Groq"
            })

    return perguntas[:quantidade]
