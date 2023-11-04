import streamlit as st  # Import the Streamlit framework for building web apps
from dotenv import load_dotenv  # Import the dotenv module for loading environment variables
import pickle  # Import the pickle module for serializing and deserializing Python objects
from PyPDF2 import PdfReader  # Import PyPDF2 for working with PDF files
from streamlit_extras.add_vertical_space import add_vertical_space  # Import a custom Streamlit component for adding vertical space
from langchain.text_splitter import RecursiveCharacterTextSplitter  # Import a text splitter from LangChain
from langchain.embeddings.openai import OpenAIEmbeddings  # Import OpenAI embeddings from LangChain
from langchain.vectorstores import FAISS  # Import FAISS for storing and retrieving vectors
from langchain.llms import OpenAI  # Import OpenAI LLM from LangChain
from langchain.chains.question_answering import load_qa_chain  # Import a question answering chain from LangChain
from langchain.callbacks import get_openai_callback  # Import a callback function for OpenAI
import os  # Import the 'os' module for interacting with the operating system

class ConversationBuffer:
    def __init__(self):
        self.buffer = []  # Initialize an empty list to store conversation messages

    def add_message(self, message):
        self.buffer.append(message)  # Add a message to the conversation buffer

    def get_messages(self):
        return self.buffer  # Retrieve all messages in the conversation buffer

    def clear_buffer(self):
        self.buffer = []  # Clear the conversation buffer

# Initialize a conversation buffer
conversation_buffer = ConversationBuffer()

# Sidebar contents
with st.sidebar:  # Define the content for the sidebar
    st.title('LLM Chat App')  # Set the title for the sidebar
    st.markdown('''
    ## About
    This app is an LLM powered chatbot using:
    - [Streamlit](https://streamlit.io/)
    - [LangChain](https://python.langchain.com/)
    - [OpenAI](https://platform.openai.com/docs/models) LLM model                
                
    ''')  # Display information about the app using markdown
    add_vertical_space(5)  # Add vertical space between elements
    st.write('Crafted with ✨ by [Prince Choudhury](https://www.linkedin.com/in/prince-choudhury26/)') 
   # Display the name of the creator

load_dotenv()  # Load environment variables from a .env file 

def main():
    st.header("Chat with PDF 📚")  # Display the main header of the app


    # Upload a PDF file
    pdf = st.file_uploader("Upload your PDF", type='pdf')  # Create a file uploader for PDF files

    if pdf is not None:  # Check if a PDF file is uploaded
        pdf_reader = PdfReader(pdf)  # Create a PdfReader object from the uploaded PDF
        st.write(pdf.name)  # Display the name of the uploaded PDF

        text = ""  # Initialize an empty string to store the text from the PDF
        for page in pdf_reader.pages:  # Loop through each page in the PDF
            text += page.extract_text()  # Extract text from the page and append it to the text string

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,  # Set the chunk size for text splitting
            chunk_overlap=200,  # Set the overlap size for consecutive chunks
            length_function=len  # Set the length function for text splitting
        )
        chunks = text_splitter.split_text(text=text)  # Split the text into chunks

        # Embeddings
        store_name = pdf.name[:-4]  # Get the name of the PDF file without the extension

        if os.path.exists(f"{store_name}.pk1"):  # Check if the vector store file exists
            with open(f"{store_name}.pk1", "rb") as f:  # Open the vector store file for reading
                VectorStore = pickle.load(f)  # Load the vector store from the file using pickle
        else:  # If the vector store file does not exist
            embeddings = OpenAIEmbeddings()  # Create OpenAI embeddings
            VectorStore = FAISS.from_texts(chunks, embedding=embeddings)  # Create a vector store from the text chunks
            with open(f"{store_name}.pk1", "wb") as f:  # Open the vector store file for writing
                pickle.dump(VectorStore, f)  # Save the vector store to the file using pickle

    # Accept user questions/queries
    query = st.text_input("Ask questions about your PDF file:")  # Create a text input for user queries
    st.write(query)  # Display the user's query

    if query:  # Check if the user has entered a query
        docs = VectorStore.similarity_search(query=query)  # Search for similar documents in the vector store

        # Create an instance of OpenAI
        llm = OpenAI()  # Create an instance of the OpenAI class

        # Specify the chain type without an extra field
        chain = load_qa_chain(llm=llm)  # Load a question answering chain using OpenAI

        with get_openai_callback() as cb:  # Create a callback for OpenAI
            response = chain.run(input_documents=docs, question=query)  # Run the question answering chain
            conversation_buffer.add_message(("User", query))  # Add the user's query to the conversation buffer
            conversation_buffer.add_message(("Chatbot", response))  # Add the chatbot's response to the conversation buffer

        st.write(response)  # Display the chatbot's response

    # Display conversation history
    conversation_history = conversation_buffer.get_messages()  # Get the list of messages from the conversation buffer
    for role, message in conversation_history:  # Loop through the messages in the conversation history
        if role == "User":  # Check if the message is from the user
            st.text_input("User:", message, key=message)  # Display the user's message
        elif role == "Chatbot":  # Check if the message is from the chatbot
            st.text_input("Chatbot:", message, key=message)  # Display the chatbot's message

if __name__ == '__main__':
    main()  # Call the main function when the script is executed
