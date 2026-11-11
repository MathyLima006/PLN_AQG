
import streamlit as st
import pandas as pd

# ==========================================
# CONFIGURAÇÃO
# ==========================================

st.set_page_config(
    page_title="Análise | AQG Lab",
    page_icon="🔎",
    layout="wide"
)

st.title("🔎 Análise dos Métodos")

st.write(
    "Conheça os fundamentos das técnicas de PLN, "
    "analise seu desempenho e avalie a qualidade "
    "das perguntas geradas."
)

st.divider()


# ==========================================
# INFORMAÇÕES SOBRE OS MÉTODOS
# ==========================================

METODOS = {
    "Regras linguísticas": {
        "categoria": "Abordagem simbólica",
        "funcionamento": (
            "Utiliza expressões regulares para identificar "
            "padrões linguísticos, como definições e autoria. "
            "Quando um padrão é encontrado, o sistema "
            "constrói uma pergunta a partir da frase."
        ),
        "exemplo": (
            "Texto: Python é uma linguagem de programação.\n\n"
            "Pergunta: O que é Python?"
        ),
        "vantagens": [
            "Execução rápida",
            "Resultados previsíveis",
            "Fácil interpretação e manutenção"
        ],
        "limitacoes": [
            "Depende de regras previamente definidas",
            "Tem dificuldade com frases complexas",
            "Não realiza compreensão semântica profunda"
        ]
    },

    "TF-IDF": {
        "categoria": "Abordagem estatística",
        "funcionamento": (
            "Calcula a importância relativa dos termos "
            "utilizando TF-IDF. Em seguida, seleciona "
            "uma palavra relevante de cada frase e "
            "a substitui por uma lacuna."
        ),
        "exemplo": (
            "Texto: Python é utilizado na ciência de dados.\n\n"
            "Possível pergunta: Complete: Python é utilizado "
            "na ______ de dados."
        ),
        "vantagens": [
            "Baixo custo computacional",
            "Não necessita de API externa",
            "Permite identificar termos relevantes"
        ],
        "limitacoes": [
            "Não compreende o contexto semântico",
            "Pode selecionar palavras inadequadas",
            "Produz perguntas principalmente de lacunas"
        ]
    },

    "Transformer T5": {
        "categoria": "Aprendizado profundo — Seq2Seq",
        "funcionamento": (
            "Utiliza uma arquitetura Transformer "
            "encoder-decoder. O modelo recebe uma "
            "sequência de entrada e produz outra "
            "sequência, permitindo formular perguntas "
            "em linguagem natural. Nossa implementação "
            "utiliza um modelo ajustado para português."
        ),
        "exemplo": (
            "Texto: Python foi criado por Guido van Rossum.\n\n"
            "Possível pergunta: Quem criou Python?"
        ),
        "vantagens": [
            "Geração de linguagem natural",
            "Utiliza representações neurais do texto",
            "Pode funcionar localmente"
        ],
        "limitacoes": [
            "Maior tempo de processamento",
            "Depende da qualidade do treinamento",
            "Pode produzir perguntas ou respostas incorretas"
        ]
    },

    "Groq": {
        "categoria": "Modelo generativo de linguagem via API",
        "funcionamento": (
            "Utiliza a plataforma Groq para executar "
            "um modelo de linguagem de grande porte. "
            "No projeto, utilizamos o GPT-OSS 20B, "
            "orientado por instruções para gerar "
            "perguntas em português, com dificuldade "
            "e formato configuráveis."
        ),
        "exemplo": (
            "Texto: Python foi criado por Guido van Rossum.\n\n"
            "Possível pergunta: Quem foi o criador "
            "da linguagem Python?"
        ),
        "vantagens": [
            "Flexibilidade na geração",
            "Permite diferentes formatos e dificuldades",
            "Não exige executar o modelo no computador"
        ],
        "limitacoes": [
            "Depende de conexão com a internet",
            "Está sujeito aos limites da API",
            "Pode gerar informações não presentes no texto"
        ]
    }
}


# ==========================================
# FUNDAMENTAÇÃO TÉCNICA
# ==========================================

st.header("📚 Fundamentos das técnicas")

abas = st.tabs(list(METODOS.keys()))

for aba, (nome, dados) in zip(abas, METODOS.items()):

    with aba:

        st.subheader(nome)
        st.caption(dados["categoria"])

        st.markdown("#### Como funciona?")
        st.write(dados["funcionamento"])

        st.markdown("#### Exemplo ilustrativo")
        st.info(dados["exemplo"])

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### Vantagens")

            for vantagem in dados["vantagens"]:
                st.write(f"✅ {vantagem}")

        with col2:
            st.markdown("#### Limitações")

            for limitacao in dados["limitacoes"]:
                st.write(f"⚠️ {limitacao}")


st.divider()


# ==========================================
# RESULTADOS COMPARTILHADOS
# ==========================================

st.header("📊 Análise de desempenho")

resultados = st.session_state.get(
    "resultados_metodos", {}
)

if not resultados:

    st.info(
        "Nenhuma execução encontrada. Gere perguntas "
        "na página principal para visualizar os dados."
    )

else:

    registros = []

    for nome, resultado in resultados.items():

        registros.append({
            "Técnica": nome,
            "Perguntas": len(resultado.get("perguntas", [])),
            "Tempo (s)": round(
                resultado.get("tempo", 0), 3
            ),
            "Status": (
                "Erro" if resultado.get("erro")
                else "Concluído"
            )
        })

    df = pd.DataFrame(registros)

    concluidos = df[
        df["Status"] == "Concluído"
    ].copy()

    st.dataframe(
        df,
        hide_index=True,
        use_container_width=True
    )

    if not concluidos.empty:

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                "Métodos concluídos",
                len(concluidos)
            )

        with col2:
            st.metric(
                "Perguntas geradas",
                int(concluidos["Perguntas"].sum())
            )

        with col3:
            st.metric(
                "Tempo acumulado",
                f"{concluidos['Tempo (s)'].sum():.2f} s"
            )

        col_g1, col_g2 = st.columns(2)

        with col_g1:
            st.subheader("Perguntas geradas")

            st.bar_chart(
                concluidos.set_index("Técnica")["Perguntas"]
            )

        with col_g2:
            st.subheader("Tempo de execução")

            st.bar_chart(
                concluidos.set_index("Técnica")["Tempo (s)"]
            )

        st.caption(
            "Os tempos incluem carregamento dos modelos "
            "e comunicação com APIs, quando aplicável."
        )


st.divider()


# ==========================================
# AVALIAÇÃO QUALITATIVA
# ==========================================

st.header("📝 Avaliação qualitativa")

st.write(
    "Avalie as perguntas considerando três critérios:"
)

st.markdown(
    """
    - **Clareza:** a pergunta é fácil de compreender?
    - **Relevância:** a pergunta aborda uma informação importante?
    - **Correção:** a pergunta e sua resposta estão corretas em relação ao texto?
    """
)

st.caption(
    "Utilize notas de 1 a 5. "
    "As avaliações são atribuídas manualmente."
)


# ==========================================
# ARMAZENAMENTO DAS AVALIAÇÕES
# ==========================================

if "avaliacoes_aqg" not in st.session_state:
    st.session_state.avaliacoes_aqg = {}

perguntas_disponiveis = {}

for nome, resultado in resultados.items():

    if resultado.get("erro"):
        continue

    perguntas = resultado.get("perguntas", [])

    if perguntas:
        perguntas_disponiveis[nome] = perguntas


if not perguntas_disponiveis:

    st.info(
        "Gere perguntas na página principal para "
        "habilitar a avaliação."
    )

else:

    metodo = st.selectbox(
        "Selecione o método:",
        list(perguntas_disponiveis.keys())
    )

    perguntas = perguntas_disponiveis[metodo]

    st.subheader(f"Avaliando: {metodo}")

    for indice, item in enumerate(perguntas, 1):

        # Identificador associado ao conteúdo.
        identificador = (
            metodo,
            item["pergunta"],
            item["resposta"]
        )

        with st.container(border=True):

            st.markdown(f"### Pergunta {indice}")

            st.write(item["pergunta"])

            st.markdown(
                f"**Resposta:** {item['resposta']}"
            )

            with st.form(
                key=f"form_{metodo}_{indice}"
            ):

                col1, col2, col3 = st.columns(3)

                with col1:
                    clareza = st.slider(
                        "Clareza",
                        min_value=1,
                        max_value=5,
                        value=3,
                        key=f"cl_{metodo}_{indice}"
                    )

                with col2:
                    relevancia = st.slider(
                        "Relevância",
                        min_value=1,
                        max_value=5,
                        value=3,
                        key=f"rel_{metodo}_{indice}"
                    )

                with col3:
                    correcao = st.slider(
                        "Correção",
                        min_value=1,
                        max_value=5,
                        value=3,
                        key=f"cor_{metodo}_{indice}"
                    )

                salvar = st.form_submit_button(
                    "💾 Salvar avaliação",
                    use_container_width=True
                )

                if salvar:

                    media = (
                        clareza + relevancia + correcao
                    ) / 3

                    st.session_state.avaliacoes_aqg[
                        identificador
                    ] = {
                        "Técnica": metodo,
                        "Pergunta": item["pergunta"],
                        "Clareza": clareza,
                        "Relevância": relevancia,
                        "Correção": correcao,
                        "Média": round(media, 2)
                    }

                    st.success("Avaliação salva!")


# ==========================================
# RESUMO DAS AVALIAÇÕES
# ==========================================

avaliacoes = list(
    st.session_state.avaliacoes_aqg.values()
)

if avaliacoes:

    st.divider()
    st.header("📈 Resultados das avaliações")

    df_avaliacoes = pd.DataFrame(avaliacoes)

    st.dataframe(
        df_avaliacoes,
        hide_index=True,
        use_container_width=True
    )

    resumo = (
        df_avaliacoes.groupby("Técnica")
        .agg(
            Media_qualidade=("Média", "mean"),
            Perguntas_avaliadas=("Média", "count")
        )
        .reset_index()
    )

    resumo["Media_qualidade"] = (
        resumo["Media_qualidade"].round(2)
    )

    st.subheader("Qualidade média por técnica")

    st.bar_chart(
        resumo.set_index("Técnica")["Media_qualidade"]
    )

    st.dataframe(
        resumo.rename(columns={
            "Media_qualidade": "Média de qualidade",
            "Perguntas_avaliadas": "Perguntas avaliadas"
        }),
        hide_index=True,
        use_container_width=True
    )

    st.caption(
        "As médias consideram somente perguntas "
        "avaliadas manualmente. As avaliações ficam "
        "na sessão atual do Streamlit e não são "
        "salvas permanentemente em banco de dados."
    )

else:

    st.info(
        "Nenhuma avaliação foi salva ainda."
    )


# ==========================================
# CONCLUSÃO
# ==========================================

st.divider()

st.header("🎯 Considerações finais")

st.write(
    "As quatro abordagens apresentam características "
    "diferentes. Regras linguísticas priorizam "
    "previsibilidade; TF-IDF identifica termos "
    "estatisticamente relevantes; T5 utiliza "
    "aprendizado profundo; e o GPT-OSS 20B, "
    "acessado pela Groq, utiliza geração de "
    "linguagem orientada por instruções."
)

st.write(
    "A escolha do melhor método depende do objetivo. "
    "Velocidade, quantidade e qualidade das perguntas "
    "devem ser consideradas em conjunto."
)

st.caption(
    "AQG Lab | Análise de técnicas de PLN"
)
