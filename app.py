
import streamlit as st
from dotenv import load_dotenv
import os
from langchain_groq import ChatGroq
import time 
from langchain_google_genai import ChatGoogleGenerativeAI
# from langchain_anthropic import ChatAnthropic


load_dotenv()

# ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")


mixtral_model = ChatGroq(model="mixtral-8x7b-32768")
gemini_model = ChatGoogleGenerativeAI(model="gemini-1.5-flash")

# claude_model = ChatAnthropic(model="claude-3-5-sonnet-20240620")

st.title("🌐 ModelBlend : Combining Diverse AI Models for the Best Outcome")
st.markdown("LLMs that you can use : Gemini-1.5-flash, gpt-3.5, llama-1, mixtral-7B")

query = st.text_area("Enter your query: ")

submit = st.button("submit")

def otpt(model):
    output = model.invoke(query).content
    return output


def get_model_output(model):
    start_time = time.time()
    output = model.invoke(query).content
    end_time = time.time()
    time_taken = end_time - start_time
    return output, time_taken


if submit:
    # Getting output and time taken by Mixtral model
    mixtral_output, mixtral_time_taken = get_model_output(mixtral_model)
    st.write(f"Mixtral Model Output:\n{mixtral_output}")
    st.write(f"Time taken by Mixtral Model: {mixtral_time_taken:.4f} seconds")

    st.markdown("------")

    # Getting output and time taken by Gemini pro model
    gemini_output, gemini_time_taken = get_model_output(gemini_model)
    st.write(f"Gemini Model Output:\n{gemini_output}")
    st.write(f"Time taken by Gemini Model: {gemini_time_taken:.4f} seconds")



    st.markdown("------")

    #Getting output and time taken by Gemini flash model
    # claude_output, claude_time_taken = get_model_output(claude_model)
    # st.write(f"Gemini Model Output:\n{claude_output}")
    # st.write(f"Time taken by Gemini Model: {claude_time_taken:.4f} seconds")

