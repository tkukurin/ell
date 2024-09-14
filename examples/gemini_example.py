"""Modified version, experimenting to get Gemini to work.
"""
import ell

# import os
# import google.generativeai as genai
# genai.configure(api_key=os.environ["GEMINI_API_KEY"])

@ell.simple(model="gemini-1.5-flash", temperature=0.1, n=1)
def number_to_words(number: int):
    """You are an expert in the english language and convert any number to its word representation, for example 123456 would be one hundred and twenty three thousand four hundred fifty six. 
You must always return the word representation and nothing else."""
    return f"Convert {number} to its word representation."

print(number_to_words("6123"))
