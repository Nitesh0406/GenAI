import os
from langchain_openai import ChatOpenAI
 

def get_supervisor_llm(api_key: str = None):
    """
    Fetches the LLM instance dynamically, ensuring it is always up-to-date.
    """
    try:
        if api_key is None:
            api_key = os.getenv("OPENAI_API_KEY", "")

        if not api_key:
            raise ValueError("⚠️ OpenAI API Key is missing! Set it via argument or 'OPENAI_API_KEY' env variable.")
        llm = ChatOpenAI(model="gpt-4o-mini", api_key=api_key, temperature=0)
        return llm
    except Exception as e:
        raise RuntimeError(f"LLM initialization failed: {str(e)}")