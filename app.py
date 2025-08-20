import streamlit as st
from summarizer import TextSummarizer, DEFAULT_MODEL

st.set_page_config(page_title="Resumos de Textos", page_icon="📝", layout="wide")
st.title("📝 Resumos de Textos (Streamlit)")
st.write("Cole um texto grande ou faça upload de um `.txt` para gerar um resumo automaticamente.")

with st.sidebar:
    st.header("Configurações")
    model_name = st.text_input("Modelo (Transformers)", value=DEFAULT_MODEL)
    max_summary_tokens = st.slider("Tamanho máx. do resumo (tokens)", 60, 300, 140, step=10)
    min_summary_tokens = st.slider("Tamanho mín. do resumo (tokens)", 10, 120, 40, step=5)
    recursive = st.toggle("Consolidação em 2 passos", value=True)

uploaded = st.file_uploader("Upload de arquivo .txt (opcional)", type=["txt"])
text_default = uploaded.read().decode("utf-8", errors="ignore") if uploaded else ""

text = st.text_area("Texto de entrada", value=text_default, height=300, placeholder="Cole aqui o texto a ser resumido...")

if st.button("Gerar Resumo", type="primary"):
    if not text.strip():
        st.warning("Por favor, informe um texto ou faça upload de um arquivo .txt.")
    else:
        with st.spinner("Carregando modelo e gerando o resumo..."):
            try:
                summarizer = TextSummarizer(model_name=model_name, device=-1)
                summary = summarizer.summarize(text, max_summary_tokens=max_summary_tokens, min_summary_tokens=min_summary_tokens, recursive=recursive)
                st.subheader("Resumo")
                st.write(summary)
                st.download_button("Baixar resumo (.txt)", data=summary, file_name="resumo.txt", mime="text/plain")
            except Exception as e:
                st.error(f"Ocorreu um erro: {e}")
                st.info("Dica: use o modelo padrão ou reduza os tamanhos de resumo.")
