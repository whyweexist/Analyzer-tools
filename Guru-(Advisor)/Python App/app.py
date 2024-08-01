import streamlit as st 
from constants.SideBar import description,title
from constants.GPT import apiKey,openaiModel
from openai import OpenAI



if __name__ == '__main__':

    client = OpenAI(api_key=apiKey)
    st.title(title)

    st.sidebar.title(title)
    st.sidebar.markdown(description)

    if "openai_model" not in st.session_state:
        st.session_state["openai_model"] = openaiModel
    
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])        


    
    if prompt := st.chat_input("Feel free to ask queries based on your finance profile.😊"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

    # with st.chat_message("assistant"):
    #     try:
    #         stream = client.chat.completions.create(
    #             model=st.session_state["openai_model"],
    #             messages=[
    #                 {"role": m["role"], "content": m["content"]}
    #                 for m in st.session_state.messages
    #             ],
    #             stream=True,
    #         )
    #         response = st.write_stream(stream)
    #     except:
    #         response = None 
    # if response:
    st.session_state.messages.append({"role": "assistant", "content": "Echo : " + str(prompt)})

    