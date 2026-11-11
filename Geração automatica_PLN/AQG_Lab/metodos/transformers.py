
import re
import unicodedata
from functools import lru_cache

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


MODELO = "vabatista/question-generation-t5-small-pt-br"


# ==========================================
# CARREGAMENTO DO MODELO
# ==========================================

@lru_cache(maxsize=1)
def carregar_modelo():
    tokenizer = AutoTokenizer.from_pretrained(MODELO)
    modelo = AutoModelForSeq2SeqLM.from_pretrained(MODELO)
    modelo.eval()
    return tokenizer, modelo


# ==========================================
# NORMALIZAÇÃO E LIMPEZA
# ==========================================

def normalizar(texto):
    texto = unicodedata.normalize("NFKD", texto)
    texto = "".join(
        c for c in texto
        if not unicodedata.combining(c)
    )
    return re.sub(r"\s+", " ", texto.casefold()).strip()


def limpar_respostas(saida):
    """
    Remove marcadores de separação e retorna
    possíveis respostas extraídas pelo T5.
    """
    saida = re.sub(
        r"</?s>|<pad>|<extra_id_\d+>",
        "",
        saida
    )

    partes = re.split(
        r"<sep>|<separator>",
        saida,
        flags=re.IGNORECASE
    )

    return [
        parte.strip(" \n\t.,;:")
        for parte in partes
        if parte.strip(" \n\t.,;:")
    ]


def dividir_frases(texto):
    frases = re.split(
        r"(?<=[.!?])\s+|\n+",
        texto.strip()
    )

    return [
        frase.strip()
        for frase in frases
        if len(frase.split()) >= 4
    ]


def localizar_resposta(frase, resposta):
    """
    Localiza a resposta na frase, preservando
    o texto original e aceitando diferenças
    de acentuação e espaços.
    """
    correspondencia = re.search(
        re.escape(resposta),
        frase,
        flags=re.IGNORECASE
    )

    if correspondencia:
        return correspondencia.span()

    # Mapeia caracteres normalizados para
    # posições do texto original.
    texto_normalizado = ""
    indices = []

    for i, caractere in enumerate(frase):
        parte = normalizar(caractere)

        for c in parte:
            texto_normalizado += c
            indices.append(i)

    resposta_normalizada = normalizar(resposta)

    posicao = texto_normalizado.find(resposta_normalizada)

    if posicao < 0 or not resposta_normalizada:
        return None

    inicio = indices[posicao]
    fim = indices[
        posicao + len(resposta_normalizada) - 1
    ] + 1

    return inicio, fim


# ==========================================
# GERAÇÃO COM O T5
# ==========================================

def gerar_sequencia(
    tokenizer,
    modelo,
    entrada,
    max_novos_tokens=64,
    amostragem=False
):
    entradas = tokenizer(
        entrada,
        return_tensors="pt",
        truncation=True,
        max_length=384
    )

    parametros = {
        "max_new_tokens": max_novos_tokens
    }

    if amostragem:
        parametros.update({
            "do_sample": True,
            "temperature": 0.85,
            "top_p": 0.92
        })
    else:
        parametros.update({
            "num_beams": 3,
            "early_stopping": True
        })

    with torch.inference_mode():
        saida = modelo.generate(
            **entradas,
            **parametros
        )

    return tokenizer.decode(
        saida[0],
        skip_special_tokens=True
    ).strip()


def pergunta_valida(pergunta, existentes):
    pergunta = pergunta.strip()

    if len(pergunta.split()) < 3:
        return False

    if "<sep>" in pergunta.lower():
        return False

    chave = normalizar(pergunta).rstrip("?")

    return chave not in existentes


# ==========================================
# FUNÇÃO PRINCIPAL
# ==========================================

def gerar_perguntas_transformer(texto, quantidade=3):

    if not texto or not texto.strip():
        return []

    quantidade = max(1, int(quantidade))

    tokenizer, modelo = carregar_modelo()
    frases = dividir_frases(texto)

    perguntas = []
    existentes = set()

    for frase in frases:

        if len(perguntas) >= quantidade:
            break

        # ETAPA 1: EXTRAÇÃO DE RESPOSTAS
        saida_respostas = gerar_sequencia(
            tokenizer,
            modelo,
            f"extrair respostas: {frase}",
            max_novos_tokens=80
        )

        respostas = limpar_respostas(saida_respostas)

        # O modelo pode extrair várias respostas.
        for resposta in respostas:

            if len(perguntas) >= quantidade:
                break

            posicao = localizar_resposta(
                frase,
                resposta
            )

            if posicao is None:
                continue

            inicio, fim = posicao

            resposta_original = frase[inicio:fim]

            # ETAPA 2: DESTACAR RESPOSTA
            contexto = (
                frase[:inicio]
                + "<hl> "
                + resposta_original
                + " <hl>"
                + frase[fim:]
            )

            entrada = f"gerar pergunta: {contexto}"

            # ETAPA 3: GERAR PERGUNTA
            for tentativa in range(2):

                pergunta = gerar_sequencia(
                    tokenizer,
                    modelo,
                    entrada,
                    max_novos_tokens=64,
                    amostragem=(tentativa == 1)
                )

                if not pergunta_valida(
                    pergunta,
                    existentes
                ):
                    continue

                if not pergunta.endswith("?"):
                    pergunta += "?"

                chave = normalizar(pergunta).rstrip("?")
                existentes.add(chave)

                perguntas.append({
                    "pergunta": pergunta,
                    "resposta": resposta_original,
                    "regra": "Transformer T5 — Seq2Seq"
                })

                break

    return perguntas[:quantidade]
