import streamlit as st
import google.generativeai as genai
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.vectorstores import FAISS
from google.generativeai.types import HarmCategory, HarmBlockThreshold


def user_input(user_question):

    prompt_template = """   
    Your are a friendly chatbot named JURIS(Judicial Understanding & Response Intelligent System) who gives information on crime, legal and law knowledge who also tell the user to follow the law
    you are supposed to give answers related to given context and analyze the question correctly and then give answer.If the question is related to any coding part, dont't give answer.
    Context: {context}?
    
    Question: {question}
    """
    embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001", google_api_key="AIzaSyDmFYX77xebjkZppD7FzBtf3qqqrlmAeGo")
    
    new_db = FAISS.load_local("faiss_index", embeddings)
    docs = new_db.similarity_search(user_question)
    print(docs)
    context_str = '\n'.join(doc.page_content for doc in docs)
    #chain = get_conversational_chain()
    promt = prompt_template.format(context = context_str, question = user_question)
    model = genai.GenerativeModel('gemini-pro', safety_settings={
        HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
    })
    response = model.generate_content(promt)
    #print(response.text)
    print(response.parts)
    return response

# Set Streamlit page configuration
st.set_page_config(page_title="JURIS")

#Title of the app
st.title("JURIS")

#Initialize chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Function to concatenate chat history into a single string
def concatenate_chat_history(chat_history):
    concatenated_history = ""
    for message in chat_history:
        if isinstance(message, dict):  # Check if message is a dictionary
            if message.get("role") == "user":
                concatenated_history += f"user: {message.get('content')}\n"
            elif message.get("role") == "assistant":
                concatenated_history += f"assistant: {message.get('content')}\n"
    return concatenated_history.strip()

# Store LLM generated responses
if "messages" not in st.session_state.keys():
    st.session_state.messages = [{"role": "assistant", "content": "How may I assist you today?"}]

# Display or clear chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

#def clear_chat_history():
#    st.session_state.messages = [{"role": "assistant", "content": "How may I assist you today?"}]
#st.sidebar.button('Clear Chat History', on_click=clear_chat_history)


# User-provided prompt
if prompt := st.chat_input():
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

# Generate a new response if last message is not from assistant
if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            #st.session_state.messages.append(prompt)
            history = concatenate_chat_history(st.session_state.messages)
            response = user_input(history)
            placeholder = st.empty()
            full_response = ''
            for item in response:
                full_response += item.text
                placeholder.markdown(full_response)
            placeholder.markdown(full_response)
    message = {"role": "assistant", "content": full_response}
    st.session_state.messages.append(message)

