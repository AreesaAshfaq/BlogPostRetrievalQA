import streamlit as st
from sentence_transformers import SentenceTransformer
from langchain import hub
from langchain_chroma import Chroma
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_text_splitters import RecursiveCharacterTextSplitter
import bs4
import torch
from langchain_groq import ChatGroq

# APP Title
st.title("Blog Retrieval and Question Answering")

# Prompt the user to enter their Langchain API key
api_key_langchain = st.text_input("Enter your LANGCHAIN_API_KEY", type="password")

# Prompt the user to enter their Groq API key
api_key_Groq = st.text_input("Enter your Groq_API_KEY", type="password")

# Check if both API keys have been provided
if not api_key_langchain or not api_key_Groq:
    st.write("Please enter both API keys to access this APP.")
else:
    st.write("Both API keys are set.")

    # Initialize the LLM with the provided Groq API key
    llm = ChatGroq(model="llama3-8b-8192", groq_api_key=api_key_Groq)

    # Define the embedding class
    class SentenceTransformerEmbedding:
        def __init__(self, model_name):
            self.model = SentenceTransformer(model_name)
        
        def embed_documents(self, texts):
            embeddings = self.model.encode(texts, convert_to_tensor=True)
            if isinstance(embeddings, torch.Tensor):
                return embeddings.cpu().detach().numpy().tolist()  # Convert tensor to list
            return embeddings
        
        def embed_query(self, query):
            embedding = self.model.encode([query], convert_to_tensor=True)
            if isinstance(embedding, torch.Tensor):
                return embedding.cpu().detach().numpy().tolist()[0]  # Convert tensor to list
            return embedding[0]

    # Initialize the embedding class
    embedding_model = SentenceTransformerEmbedding('all-MiniLM-L6-v2')

    # Streamlit UI for blog URL input
    blog_url = st.text_input("Enter the URL of the blog to retrieve:")

    # Load, chunk, and index the contents of the blog
    def load_data(url):
          try:
              loader = WebBaseLoader(
                  web_paths=(url,),
                  bs_kwargs=dict(
                      parse_only=bs4.SoupStrainer(
                      )
                  ),
              )
              docs = loader.load()
            
              # Debugging output
              #st.write(f"Loaded {len(docs)} documents from the URL.")
        
              if not docs:
                  st.error("No documents were loaded. Please check the URL or content.")
                  return None
            
              # Check the first document's content to ensure it's loaded correctly
              #st.write(f"First document content preview: {docs[0].page_content[:500]}")  # Show the first 500 characters of the first document
        
              text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
              splits = text_splitter.split_documents(docs)
            
              # Debugging output
              #st.write(f"Created {len(splits)} document splits.")
        
              if not splits:
                  st.error("No document splits were created. Please check the document content.")
                  return None
            
              # Check the first split's content to ensure it's split correctly
              #st.write(f"First split content preview: {splits[0].page_content[:500]}")  # Show the first 500 characters of the first split
        
              vectorstore = Chroma.from_documents(documents=splits, embedding=embedding_model)
            
              # Debugging output
              #st.write(f"Vectorstore created with {len(splits)} documents.")
            
              if vectorstore is None:
                  st.error("Failed to create the vectorstore.")
                  return None
            
              return vectorstore
          except Exception as e:
            st.error(f"An error occurred while loading the blog: {e}")
            return None

    # def load_data(url):
    #     try:
    #         loader = WebBaseLoader(
    #             web_paths=(url,),
    #             bs_kwargs=dict(
    #                 parse_only=bs4.SoupStrainer(
    #                     class_=("post-content", "post-title", "post-header")
    #                 )
    #             ),
    #         )
    #         docs = loader.load()
    #         text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    #         splits = text_splitter.split_documents(docs)
    #         vectorstore = Chroma.from_documents(documents=splits, embedding=embedding_model)
    #         return vectorstore
    #     except Exception as e:
    #         st.error(f"An error occurred while loading the blog: {e}")
    #         return None

    # Load the data if a URL is provided
    if blog_url:
        vectorstore = load_data(blog_url)
        if vectorstore:
            # Streamlit UI for question input
            question = st.text_input("Enter your question:")

            if question:
                retriever = vectorstore.as_retriever()
                prompt = hub.pull("rlm/rag-prompt", api_key=api_key_langchain)

                def format_docs(docs):
                    return "\n\n".join(doc.page_content for doc in docs)

                rag_chain = (
                    {"context": retriever | format_docs, "question": RunnablePassthrough()}
                    | prompt
                    | llm
                    | StrOutputParser()
                )

                # Example invocation
                try:
                    result = rag_chain.invoke(question)
                    st.write("Answer:", result)
                except Exception as e:
                    st.error(f"An error occurred while generating the answer: {e}")
        else:
            st.write("Failed to load the blog content. Please check the URL and try again.")