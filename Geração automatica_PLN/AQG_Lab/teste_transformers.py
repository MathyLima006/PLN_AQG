
from metodos.transformers import (
    carregar_modelo,
    gerar_sequencia,
    limpar_respostas,
    localizar_resposta,
    dividir_frases,
    gerar_perguntas_transformer
)

texto = """
O Lorem Ipsum é um texto modelo da indústria tipográfica e de impressão.
O Lorem Ipsum tem vindo a ser o texto padrão usado por estas indústrias desde o ano de 1500, quando uma misturou os caracteres de um texto para criar um espécime de livro.
Este texto não só sobreviveu 5 séculos, mas também o salto para a tipografia electrónica, mantendo-se essencialmente inalterada.
Foi popularizada nos anos 60 com a disponibilização das folhas de Letraset, que continham passagens com Lorem Ipsum, e mais recentemente com os programas de publicação como o Aldus PageMaker que incluem versões do Lorem Ipsum.
"""

tokenizer, modelo = carregar_modelo()

print("\n========== DIAGNÓSTICO ==========")

for numero, frase in enumerate(dividir_frases(texto), 1):
    print(f"\nFRASE {numero}: {frase}")

    saida = gerar_sequencia(
        tokenizer,
        modelo,
        f"extrair respostas: {frase}",
        max_novos_tokens=80
    )

    print("Extração bruta:", repr(saida))

    respostas = limpar_respostas(saida)
    print("Respostas limpas:", respostas)

    for resposta in respostas:
        posicao = localizar_resposta(frase, resposta)
        print(
            f"Resposta: {resposta!r} | "
            f"Encontrada no texto: {posicao is not None}"
        )

        if posicao is not None:
            inicio, fim = posicao

            contexto = (
                frase[:inicio]
                + "<hl> "
                + frase[inicio:fim]
                + " <hl>"
                + frase[fim:]
            )

            pergunta = gerar_sequencia(
                tokenizer,
                modelo,
                f"gerar pergunta: {contexto}"
            )

            print("Pergunta gerada:", repr(pergunta))

print("\n========== RESULTADO FINAL ==========")
print(gerar_perguntas_transformer(texto, 3))
