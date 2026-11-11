
import re
import random


def gerar_perguntas_regras(texto, quantidade):

    perguntas = []

    # Divide o texto em frases
    frases = re.split(r'(?<=[.!?])\s+', texto.strip())

    for frase in frases:

        frase = frase.strip().rstrip(".!?")

        if not frase:
            continue

        # REGRA 1: Identificação de definições
        definicao = re.fullmatch(
            r"(.+?)\s+(é|são)\s+(.+)",
            frase,
            re.IGNORECASE
        )

        if definicao:

            conceito = definicao.group(1).strip()
            resposta = definicao.group(3).strip()

            conceito = re.sub(
                r"^(o|a|os|as)\s+",
                "",
                conceito,
                flags=re.IGNORECASE
            )

            perguntas.append({
                "pergunta": f"O que é {conceito}?",
                "resposta": resposta,
                "regra": "Definição por padrão linguístico"
            })

        # REGRA 2: Identificação de autoria
        criacao = re.fullmatch(
            r"(.+?)\s+foi criad[oa]\s+por\s+(.+)",
            frase,
            re.IGNORECASE
        )

        if criacao:

            objeto = criacao.group(1).strip()
            autor = criacao.group(2).strip()

            objeto = re.sub(
                r"^(o|a|os|as)\s+",
                "",
                objeto,
                flags=re.IGNORECASE
            )

            perguntas.append({
                "pergunta": f"Quem criou {objeto}?",
                "resposta": autor,
                "regra": "Identificação de autoria"
            })

    # Remove perguntas repetidas
    perguntas = list({
        p["pergunta"].lower(): p
        for p in perguntas
    }.values())

    # Embaralha as perguntas
    random.shuffle(perguntas)

    return perguntas[:quantidade]
