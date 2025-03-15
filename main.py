import os
import streamlit as st
from langchain import OpenAI
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import UnstructuredURLLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS

# Set OpenAI API Key
os.environ['OPENAI_API_KEY'] = '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'

# Initialize OpenAI LLM
llm = OpenAI(temperature=0.6)

# Streamlit UI Setup
st.title("Article Research Tool")
st.sidebar.title("Paste URLs Here")

# URL Input
url = []
for i in range(0, 3):
    u = st.sidebar.text_input(f"URL {i+1}")
    url.append(u)

# Button to Run the Model
url_click = st.sidebar.button("Run the model")

# Define FAISS storage path
faiss_store_path = "faiss_store"

main_screen = st.empty()

# FAISS Index Creation and Saving
if url_click:
    loader = UnstructuredURLLoader(urls=url)
    main_screen.text("Loading URLs...")
    
    try:
        data = loader.load()
        main_screen.text("Data loaded successfully.")
    except Exception as e:
        st.write(f"Error loading data: {e}")
        data = []

    if data:
        # Splitting the loaded data
        url_splitter = RecursiveCharacterTextSplitter(
            separators=['\n\n', '\n', '.', ','],
            chunk_size=1000
        )
        main_screen.text("Splitting data into chunks...")
        docs = url_splitter.split_documents(data)

        # Embedding the data and saving it to a FAISS index
        main_screen.text("Generating embeddings and saving FAISS index...")
        try:
            emb = OpenAIEmbeddings()
            vector_store = FAISS.from_documents(docs, emb)

            # Save vector store using FAISS's native method
            vector_store.save_local(faiss_store_path)

            main_screen.text("FAISS index saved successfully!")
        except Exception as e:
            st.write(f"Error saving FAISS index: {e}")
    else:
        st.write("No valid data to process.")

# Question Input and Retrieval
query = main_screen.text_input("QUESTION:")

if query:
    if os.path.exists(faiss_store_path):
        try:
            main_screen.text("Loading FAISS index from directory...")
            vector_store = FAISS.load_local(faiss_store_path, OpenAIEmbeddings(), allow_dangerous_deserialization=True)

            chain = RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever=vector_store.as_retriever())

            main_screen.text("Running query...")
            result = chain({"question": query}, return_only_outputs=True)

            st.header("Answer")
            st.write(result.get("answer", "No answer found."))

            st.header("Sources")
            st.write(result.get("sources", "No sources found."))
        except Exception as e:
            st.write(f"Error during query execution: {e}")
    else:
        st.write("FAISS index not found! Please run the embedding process first.")

