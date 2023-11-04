import streamlit as st  # Import the Streamlit library for building web apps
import pickle  # Import the pickle module for object serialization
from dotenv import load_dotenv  # Import the dotenv module for loading environment variables
from streamlit_extras.add_vertical_space import add_vertical_space  # Import a Streamlit component for adding vertical space
from PyPDF2 import PdfReader  # Import PyPDF2 for PDF file handling
from langchain.text_splitter import RecursiveCharacterTextSplitter  # Import LangChain for text splitting
from langchain.embeddings.openai import OpenAIEmbeddings  # Import LangChain's OpenAI embeddings
from langchain.vectorstores import FAISS  # Import LangChain's FAISS vector store
from langchain.llms import OpenAI  # Import LangChain's OpenAI language model
from langchain.memory import ConversationBufferMemory  # Import LangChain's Conversation Buffer Memory
from langchain.chains.question_answering import load_qa_chain  # Import LangChain's question-answering chain
from langchain.callbacks import get_openai_callback  # Import LangChain's OpenAI callback
from langchain.chains import ConversationalRetrievalChain  # Import LangChain's Conversational Retrieval Chain
from langchain.prompts.prompt import PromptTemplate  # Import LangChain's prompt template
from langchain.chat_models import ChatOpenAI  # Import LangChain's ChatOpenAI
import os  # Import the os module for interacting with the operating system

# Storing chat history in session states
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state["messages"]:
    if message["role"] == "user":
        # Display a user message in the chat interface
        with st.chat_message("user"):
            st.markdown(message["content"])
    elif message["role"] == "assistant":
        # Display an assistant message in the chat interface
        with st.chat_message("assistant"):
            st.markdown(message["content"])


# Sidebar contents
with st.sidebar:
    st.title('🤖 Multi-turn ChatBot')  # Set the title for the sidebar
    st.markdown('''
    ## About
    This app is an LLM powered chatbot using:
    - [Streamlit](https://streamlit.io/)
    - [LangChain](https://python.langchain.com/)
    - [OpenAI](https://platform.openai.com/docs/models) LLM model
    ''')  # Display information about the app using markdown

    pdf = st.file_uploader("Upload your PDF", type='pdf')
    add_vertical_space(2)  # Add vertical space between elements
    st.write('Crafted with ✨ by [Prince Choudhury](https://www.linkedin.com/in/prince-choudhury26/)')

def main(pdf):
    load_dotenv()  # Load environment variables from a .env file

    st.header("Chat with PDF 📚")  # Display the main header

    if pdf is not None:
        # PDF reader
        pdf_reader = PdfReader(pdf)

        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()  # Extract text from each page and concatenate it

        # Text splitting
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,  # Set the chunk size to 1000 tokens
            chunk_overlap=200,  # Set the overlap size to 200 tokens between consecutive chunks
            length_function=len
        )
        chunks = text_splitter.split_text(text=text)  # Split the text into chunks

        # PDF embeddings
        store_name = pdf.name[:-4]  # Get the name of the PDF file without the extension

        if os.path.exists(f"{store_name}.pkl"):  # Check if the vector store file exists
            with open(f"{store_name}.pkl", "rb") as f:
                VectorStore = pickle.load(f)  # Load the vector store object from the file
        else:
            embeddings = OpenAIEmbeddings()  # Create embeddings using OpenAI
            VectorStore = FAISS.from_texts(chunks, embedding=embeddings)  # Create a vector store from the text chunks

        # Accept user questions/queries
        query = st.chat_input(placeholder="Ask questions about your PDF file:") #to create an input field where the user can ask questions about the PDF file.
        memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)  #initialize the memory using ConversationBufferMemory with a specific memory key to store chat history and return messages.

        if query:  #check if the user has entered a query. If a query is provided, we proceed with processing it.
            chat_history = [] #to store the conversation history between the user and the chatbot.
            with st.chat_message("user"):
                st.markdown(query)  # Display the user's query
            st.session_state.messages.append({"role": "user", "content": query})  # Store the user's query in the chat history

            custom_template = """
            Given the following conversation and a follow-up question, rephrase the follow-up question to be a standalone question.
            At the end of the standalone question, add this 'Answer the question in English language.'
            If you do not know the answer, reply with 'I am sorry'.
            Chat History:
            {chat_history}
            Follow Up Input: {question}
            Standalone question:
            Remember to greet the user with 'hi welcome to the PDF chatbot, how can I help you?' if the user asks 'hi' or 'hello.'
            """

            # Create a custom prompt template for generating questions based on conversation history
            CUSTOM_QUESTION_PROMPT = PromptTemplate.from_template(custom_template)

            # Create a ChatOpenAI instance to interact with the chatbot model
            llm = ChatOpenAI()

            # Create a Conversational Retrieval Chain to manage the conversation
            conversation_chain = ConversationalRetrievalChain.from_llm(
                llm,
                VectorStore.as_retriever(),
                condense_question_prompt=CUSTOM_QUESTION_PROMPT,
                memory=memory
            )

            # Get a response from the chatbot for the user's query and conversation history
            response = conversation_chain({"question": query, "chat_history": chat_history})

            # Display the chatbot's response as a message in the chat interface
            with st.chat_message("assistant"):
                st.markdown(response["answer"])

            # Store the chatbot's response in the chat history
            st.session_state.messages.append({"role": "assistant", "content": response["answer"]})

            # Append the user's query and the chatbot's response to the chat history
            chat_history.append((query, response))


if __name__ == '__main__':
    main(pdf)  
