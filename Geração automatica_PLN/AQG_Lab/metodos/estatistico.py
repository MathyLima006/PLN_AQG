
import re
import random

from sklearn.feature_extraction.text import TfidfVectorizer


# ==========================================
# GERAÇÃO DE PERGUNTAS COM TF-IDF
# ==========================================

def gerar_perguntas_tfidf(texto, quantidade):

    # Divide o texto em frases
    frases = re.split(
        r'(?<=[.!?])\s+',
        texto.strip()
    )

    # Mantém apenas frases com pelo menos 5 palavras
    frases = [
        frase.strip()
        for frase in frases
        if len(frase.split()) >= 5
    ]

    if not frases:
        return []

    # Palavras comuns da língua portuguesa
    stopwords = [
        "a", "o", "as", "os", "um", "uma",
        "de", "do", "da", "dos", "das",
        "em", "no", "na", "nos", "nas",
        "por", "para", "com", "sem",
        "que", "se", "e", "ou", "é",
        "são", "foi", "ser", "como",
        "mais", "menos", "ao", "aos",
        "sua", "seu", "suas", "seus",
        "entre", "também", "muito"
    ]

    # Calcula a relevância estatística dos termos
    vectorizer = TfidfVectorizer(
        stop_words=stopwords,
        lowercase=True
    )

    try:
        matriz = vectorizer.fit_transform(frases)

    except ValueError:
        return []

    palavras = vectorizer.get_feature_names_out()

    perguntas = []

    # Analisa cada frase
    for indice, frase in enumerate(frases):

        pontuacoes = matriz[indice].toarray()[0]

        # Ordena termos por relevância
        indices_ordenados = pontuacoes.argsort()[::-1]

        for idx in indices_ordenados:

            if pontuacoes[idx] <= 0:
                continue

            palavra = palavras[idx]

            # Localiza o termo no texto original
            padrao = r"\b" + re.escape(palavra) + r"\b"

            encontrado = re.search(
                padrao,
                frase,
                re.IGNORECASE
            )

            if encontrado:

                resposta = encontrado.group()

                # Substitui o termo por uma lacuna
                pergunta = re.sub(
                    padrao,
                    "______",
                    frase,
                    count=1,
                    flags=re.IGNORECASE
                )

                perguntas.append({
                    "pergunta": f"Complete: {pergunta}",
                    "resposta": resposta,
                    "regra": "Método estatístico TF-IDF"
                })

                break

    # Embaralha os resultados
    random.shuffle(perguntas)

    return perguntas[:quantidade]
