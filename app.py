
## LLM Leaderboard - Comparison of different LLMs
'''
1) https://artificialanalysis.ai/leaderboards/models
2) https://bigcode-bench.github.io/
3) https://klu.ai/llm-leaderboard

IMP 4) https://www.vellum.ai/llm-leaderboard#coding
'''


import streamlit as st

from langchain_groq import ChatGroq
import time 
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI

GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]
GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]

openai_model = ChatOpenAI(model="gpt-4o-mini")
gemini_model = ChatGoogleGenerativeAI(model="gemini-1.5-pro")
llama_model = ChatGroq(model="llama-3.1-70b-versatile")

st.title("🌐 ModelBlend : Combining Diverse AI Models for the Best Outcome")
st.markdown("LLMs that you can use : Gemini-1.5-flash, gpt-3.5, llama-1, mixtral-7B")


def get_model_output(model):
    start_time = time.time()
    try:
        output = model.invoke(query).content
    except Exception as e:
        output = f"An error occurred: {str(e)}"
    end_time = time.time()
    time_taken = end_time - start_time
    return output, time_taken


def final_output(model,llm_outputs):

    # messages = [
    #     (
    #         "system",
    #         "You are a helpful assistant that analyses the outputs generated by different LLM models and provide the final solution.",
    #     ),
    #     ("human", f"{llm_outputs}"),
    # ]

    messages = [
    (
        "system",
        "You are a highly skilled assistant with expertise in analyzing and synthesizing outputs generated by various large language models (LLMs). Your goal is to evaluate the provided outputs based on accuracy, relevance, and coherence, and then provide the final solution. "
    ),
    ("human", f"{llm_outputs}"),
    ]

    output = model.invoke(messages).content
    return output

query = st.text_area("Enter your query: ")

submit = st.button("submit")

if submit:

    # Getting output and time taken by OpenAI model
    openai_output, openai_time_taken = get_model_output(openai_model)
    st.write(f"OpenAI Model Output:\n{openai_output}")
    st.write(f"Time taken by OpenAI Model: {openai_time_taken:.4f} seconds")
    st.markdown("------")

    # Getting output and time taken by Gemini pro model
    gemini_output, gemini_time_taken = get_model_output(gemini_model)
    st.write(f"Gemini Model Output:\n{gemini_output}")
    st.write(f"Time taken by Gemini Model: {gemini_time_taken:.4f} seconds")
    st.markdown("------")

    #Getting output and time taken by llama 3.1 model
    llama_output, llama_time_taken = get_model_output(llama_model)
    st.write(f"llama-3.1 Model Output:\n{llama_output}")
    st.write(f"Time taken by llama-3.1 Model: {llama_time_taken:.4f} seconds")
    st.markdown("------")




    llm_outputs = f"""
    output generated by openai model : {openai_output}\n
    output generated by llama model : {llama_output}\n
    output generated by gemini model : {gemini_output}\n
    """

    st.markdown("## Final Output")
    st.markdown(final_output(openai_model,llm_outputs))


