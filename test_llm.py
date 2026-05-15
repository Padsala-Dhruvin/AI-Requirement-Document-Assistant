import os, logging, traceback
from src.llm_service import get_llm

logging.basicConfig(level=logging.DEBUG)

def main():
    llm = get_llm()
    if llm is None:
        print("Missing or invalid GOOGLE_API_KEY")
        return

    try:
        # Minimal test - create a tiny prompt and invoke the chain if available
        from langchain_core.prompts import ChatPromptTemplate
        prompt = ChatPromptTemplate.from_template("Question: {question}\nAnswer:")
        chain = prompt | llm
        resp = chain.invoke({"question": "Say hello", "context": " "})
        print("Response:", getattr(resp, 'content', resp))
    except Exception as e:
        print("LLM call failed:")
        traceback.print_exc()

if __name__ == '__main__':
    main()