
import streamlit as st
import pandas as pd

st.set_page_config(
    page_title="Comparação | AQG Lab",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Comparação entre técnicas")
st.write(
    "Compare os resultados das quatro abordagens "
    "de geração automática de perguntas, sem "
    "precisar executar os modelos novamente."
)

st.divider()

ORDEM_METODOS = [
    "Regras linguísticas",
    "TF-IDF",
    "Transformer T5",
    "Groq"
]

resultados = st.session_state.get("resultados_metodos", {})

if not resultados:
    st.info(
        "Ainda não existem resultados. Acesse a página "
        "principal, gere perguntas e depois volte aqui."
    )
    st.stop()

# ==========================================
# SELEÇÃO DOS MÉTODOS
# ==========================================

disponiveis = [
    nome for nome in ORDEM_METODOS
    if nome in resultados
]

st.header("⚙️ Configuração da comparação")

selecionados = st.multiselect(
    "Selecione os métodos para comparar:",
    options=disponiveis,
    default=disponiveis
)

if not selecionados:
    st.warning("Selecione pelo menos um método.")
    st.stop()

# ==========================================
# ORGANIZAÇÃO DOS RESULTADOS
# ==========================================

dados = []

for nome in selecionados:
    resultado = resultados[nome]

    dados.append({
        "Técnica": nome,
        "Perguntas": len(resultado.get("perguntas", [])),
        "Tempo (s)": round(resultado.get("tempo", 0), 3),
        "Status": (
            "Erro" if resultado.get("erro") else "Concluído"
        )
    })

df = pd.DataFrame(dados)
concluidos = df[df["Status"] == "Concluído"].copy()

st.divider()
st.header("📈 Resultados gerais")

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

st.subheader("Tabela comparativa")

st.dataframe(
    df,
    use_container_width=True,
    hide_index=True
)

# ==========================================
# GRÁFICOS
# ==========================================

if not concluidos.empty:

    col_g1, col_g2 = st.columns(2)

    with col_g1:
        st.subheader("Quantidade de perguntas")

        st.bar_chart(
            concluidos.set_index("Técnica")["Perguntas"]
        )

    with col_g2:
        st.subheader("Tempo de execução")

        st.bar_chart(
            concluidos.set_index("Técnica")["Tempo (s)"]
        )

    st.caption(
        "O tempo registrado inclui carregamento de "
        "modelos e chamadas de API. Os resultados "
        "representam as execuções realizadas, não "
        "uma avaliação isolada dos algoritmos."
    )

# ==========================================
# COMPARAÇÃO DAS PERGUNTAS
# ==========================================

st.divider()
st.header("📝 Perguntas produzidas")

st.write(
    "Observe como cada método transforma "
    "o mesmo texto em perguntas."
)

colunas = st.columns(
    min(len(selecionados), 2),
    gap="large"
)

for indice, nome in enumerate(selecionados):

    with colunas[indice % len(colunas)]:

        with st.container(border=True):

            st.subheader(nome)

            resultado = resultados[nome]

            if resultado.get("erro"):
                st.error(resultado["erro"])
                continue

            perguntas = resultado.get("perguntas", [])

            if not perguntas:
                st.warning("Nenhuma pergunta disponível.")
                continue

            for numero, item in enumerate(perguntas, 1):

                st.markdown(f"**Pergunta {numero}**")
                st.write(item.get("pergunta", ""))

                st.markdown(
                    f"**Resposta:** {item.get('resposta', '')}"
                )

                if numero < len(perguntas):
                    st.divider()

# ==========================================
# OBSERVAÇÕES
# ==========================================

st.divider()

st.header("💡 Interpretação")

st.write(
    "A quantidade de perguntas mostra a capacidade "
    "de cada método de produzir resultados para o "
    "texto fornecido. O tempo indica a duração "
    "registrada durante a execução."
)

st.write(
    "Entretanto, produzir mais perguntas ou executar "
    "mais rapidamente não significa necessariamente "
    "gerar perguntas melhores. A qualidade deve ser "
    "avaliada considerando clareza, relevância e "
    "correção das informações."
)

st.caption(
    "AQG Lab | Comparação de técnicas de PLN"
)
