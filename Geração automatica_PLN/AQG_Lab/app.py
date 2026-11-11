
import time

import streamlit as st

from metodos.regras import gerar_perguntas_regras
from metodos.estatistico import gerar_perguntas_tfidf


# ==========================================
# CONFIGURAÇÃO DA PÁGINA
# ==========================================

st.set_page_config(
    page_title="AQG Lab",
    page_icon="🧠",
    layout="wide"
)

st.title("🧠 AQG Lab")
st.subheader("Laboratório de Geração Automática de Perguntas")

st.write(
    "Explore diferentes técnicas de Processamento de "
    "Linguagem Natural para gerar perguntas a partir "
    "de textos em português."
)

st.divider()


# ==========================================
# ESTADO DA APLICAÇÃO
# ==========================================

if "resultados_metodos" not in st.session_state:
    st.session_state.resultados_metodos = {}

if "texto_resultados" not in st.session_state:
    st.session_state.texto_resultados = None

if "config_resultados" not in st.session_state:
    st.session_state.config_resultados = {}


# ==========================================
# TEXTO DE ENTRADA
# ==========================================

st.header("📄 Texto de entrada")

texto = st.text_area(
    "Insira o texto que será utilizado pelos métodos:",
    height=200,
    placeholder=(
        "Exemplo: Python é uma linguagem de programação. "
        "Python foi criado por Guido van Rossum."
    )
)

quantidade = st.slider(
    "Quantidade máxima de perguntas por método",
    min_value=1,
    max_value=10,
    value=3
)

# Limpa resultados quando o texto muda.
if st.session_state.texto_resultados != texto:
    st.session_state.resultados_metodos = {}
    st.session_state.config_resultados = {}
    st.session_state.texto_resultados = texto

st.divider()


# ==========================================
# EXECUÇÃO INDIVIDUAL DOS MÉTODOS
# ==========================================

def executar_metodo(nome, funcao, configuracao=None):

    if not texto.strip():
        st.warning("Insira um texto antes de gerar perguntas.")
        return

    inicio = time.perf_counter()

    try:

        with st.spinner(f"Executando {nome}..."):
            perguntas = funcao()

        tempo = time.perf_counter() - inicio

        # Garante que o resultado tenha o formato esperado.
        if not isinstance(perguntas, list):
            raise ValueError(
                "O método não retornou uma lista de perguntas."
            )

        perguntas_validas = []

        for item in perguntas:

            if not isinstance(item, dict):
                continue

            pergunta = str(item.get("pergunta", "")).strip()
            resposta = str(item.get("resposta", "")).strip()

            if pergunta and resposta:
                perguntas_validas.append({
                    "pergunta": pergunta,
                    "resposta": resposta,
                    "regra": item.get("regra", nome)
                })

        st.session_state.resultados_metodos[nome] = {
            "perguntas": perguntas_validas,
            "tempo": tempo,
            "erro": None
        }

        st.session_state.config_resultados[nome] = configuracao or {}

    except Exception as erro:

        tempo = time.perf_counter() - inicio

        st.session_state.resultados_metodos[nome] = {
            "perguntas": [],
            "tempo": tempo,
            "erro": str(erro)
        }


# ==========================================
# EXIBIÇÃO DOS RESULTADOS
# ==========================================

def mostrar_resultados(nome):

    resultado = st.session_state.resultados_metodos.get(nome)

    if resultado is None:
        st.caption("Aguardando execução.")
        return

    if resultado["erro"]:
        st.error(f"Erro: {resultado['erro']}")
        return

    perguntas = resultado["perguntas"]

    if not perguntas:
        st.warning(
            "O método foi executado, mas não gerou "
            "perguntas válidas para este texto."
        )
        return

    st.success(
        f"{len(perguntas)} pergunta(s) gerada(s)!"
    )

    st.caption(
        f"⏱️ Tempo de execução: {resultado['tempo']:.2f} s"
    )

    for indice, item in enumerate(perguntas, 1):

        with st.container(border=True):

            st.markdown(f"**Pergunta {indice}**")

            st.write(item["pergunta"])

            st.markdown(
                f"**Resposta:** {item['resposta']}"
            )

            st.caption(
                f"Técnica: {item['regra']}"
            )


# ==========================================
# PAINÉIS DAS TÉCNICAS
# ==========================================

st.header("🔬 Técnicas de geração")

col1, col2 = st.columns(2, gap="large")


# ==========================================
# 1. REGRAS LINGUÍSTICAS
# ==========================================

with col1:

    with st.container(border=True):

        st.subheader("📘 Regras linguísticas")

        st.write(
            "Identifica estruturas linguísticas, como "
            "definições e autoria, utilizando padrões "
            "predefinidos."
        )

        if st.button(
            "Gerar com regras",
            key="btn_regras",
            use_container_width=True
        ):

            executar_metodo(
                "Regras linguísticas",
                lambda: gerar_perguntas_regras(
                    texto,
                    quantidade
                ),
                {"quantidade": quantidade}
            )

        mostrar_resultados("Regras linguísticas")


# ==========================================
# 2. TF-IDF
# ==========================================

with col2:

    with st.container(border=True):

        st.subheader("📊 Método estatístico — TF-IDF")

        st.write(
            "Identifica termos relevantes por meio "
            "de pesos estatísticos e gera perguntas "
            "de preenchimento de lacunas."
        )

        if st.button(
            "Gerar com TF-IDF",
            key="btn_tfidf",
            use_container_width=True
        ):

            executar_metodo(
                "TF-IDF",
                lambda: gerar_perguntas_tfidf(
                    texto,
                    quantidade
                ),
                {"quantidade": quantidade}
            )

        mostrar_resultados("TF-IDF")


# ==========================================
# 3. TRANSFORMER T5
# ==========================================

with col1:

    with st.container(border=True):

        st.subheader("🧠 Transformer T5")

        st.write(
            "Utiliza uma arquitetura neural "
            "Transformer encoder-decoder para "
            "gerar perguntas em português."
        )

        st.caption(
            "A primeira execução pode demorar "
            "devido ao carregamento do modelo."
        )

        if st.button(
            "Gerar com Transformer T5",
            key="btn_t5",
            use_container_width=True
        ):

            def executar_t5():

                from metodos.transformers import (
                    gerar_perguntas_transformer
                )

                return gerar_perguntas_transformer(
                    texto,
                    quantidade
                )

            executar_metodo(
                "Transformer T5",
                executar_t5,
                {"quantidade": quantidade}
            )

        mostrar_resultados("Transformer T5")


# ==========================================
# 4. GROQ — MODELO DE LINGUAGEM
# ==========================================

with col2:

    with st.container(border=True):

        st.subheader("✨ Groq — GPT-OSS 20B")

        st.write(
            "Utiliza um modelo de linguagem generativo "
            "hospedado na Groq para produzir perguntas "
            "em português brasileiro."
        )

        dificuldade = st.selectbox(
            "Dificuldade das perguntas",
            ["Fácil", "Médio", "Difícil"],
            key="dificuldade_groq"
        )

        formato = st.selectbox(
            "Formato das perguntas",
            [
                "Discursivas",
                "Múltipla escolha",
                "Verdadeiro ou falso"
            ],
            key="formato_groq"
        )

        st.caption(
            "Requer conexão com a internet e "
            "uma chave GROQ_API_KEY configurada."
        )

        if st.button(
            "Gerar com Groq",
            key="btn_groq",
            use_container_width=True
        ):

            def executar_groq():

                from metodos.groq import (
                    gerar_perguntas_groq
                )

                return gerar_perguntas_groq(
                    texto=texto,
                    quantidade=quantidade,
                    dificuldade=dificuldade,
                    formato=formato
                )

            executar_metodo(
                "Groq",
                executar_groq,
                {
                    "quantidade": quantidade,
                    "dificuldade": dificuldade,
                    "formato": formato
                }
            )

        mostrar_resultados("Groq")


# ==========================================
# RESUMO DE DESEMPENHO
# ==========================================

if st.session_state.resultados_metodos:

    st.divider()

    st.header("📈 Resumo de desempenho")

    dados = []

    for nome, resultado in st.session_state.resultados_metodos.items():

        dados.append({
            "Técnica": nome,
            "Perguntas geradas": len(resultado["perguntas"]),
            "Tempo (s)": round(resultado["tempo"], 2),
            "Status": (
                "Erro" if resultado["erro"] else "Concluído"
            )
        })

    st.dataframe(
        dados,
        use_container_width=True,
        hide_index=True
    )

    st.caption(
        "Os tempos apresentados incluem o processamento "
        "e, quando necessário, o carregamento do modelo "
        "ou a comunicação com a API."
    )


# ==========================================
# RODAPÉ
# ==========================================

st.divider()

st.caption(
    "AQG Lab | Projeto acadêmico de "
    "Processamento de Linguagem Natural"
)
