import os
import re
from dotenv import load_dotenv
from google import genai
from google.genai import types
from ddgs import DDGS

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise ValueError("GEMINI_API_KEY not found. Please check your .env file.")

client = genai.Client(api_key=api_key)

SYSTEM_PROMPT = """You are a helpful research assistant with access to a search engine.

When answering, strictly follow this structure:

Thought: Explain what information is missing.
Action: search_web[query to search]
Observation: (leave for system to inject)

Rules:
1. When you get the search results in Observation, read them and formulate your Final Answer.
2. If search results don't answer the prompt, you may search ONE more time with a broader query.
3. Once you have enough facts, finish immediately with:
Final Answer: [Your complete answer citing the URLs or sources]
"""

def search_web(query: str, max_results: int = 3) -> str:
    """Takes a search query and returns titles, snippets, and URLs."""
    try:
        with DDGS() as ddgs:
            results = ddgs.text(query, max_results=max_results, backend="html")
        
        if not results:
            return "No results found."
        
        formatted_results = []
        for r in results:
            formatted_results.append(
                f"Title: {r.get('title')}\nURL: {r.get('href')}\nSnippet: {r.get('body')}"
            )
        return "\n\n".join(formatted_results)
    except Exception as e:
        return f"Error executing search: {e}"

def ask_agent(user_question: str, max_rounds: int = 5):
    # Initialize a fresh chat session for each prompt
    chat = client.chats.create(
        model="gemini-3.5-flash-lite",
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            stop_sequences=["Observation:"],
            temperature=0.2
        )
    )

    print(f"\nProcessing query...\n" + "="*50)
    current_prompt = user_question

    for round_num in range(max_rounds):
        response = chat.send_message(current_prompt)
        agent_step = response.text.strip()
        print(f"\n[Round {round_num + 1}]\n{agent_step}")

        if "Final Answer:" in agent_step:
            print("\n" + "="*50 + "\nTASK COMPLETED!")
            return agent_step.split("Final Answer:")[-1].strip()

        match = re.search(r"Action:\s*search_web\[(.*?)\]", agent_step)
        if match:
            search_query = match.group(1).strip()
            print(f"\n--> Executing Web Search: '{search_query}'")
            results = search_web(search_query)
            
            preview = results[:200] + "..." if len(results) > 200 else results
            print(f"--> Search Output Preview:\n{preview}")
            
            current_prompt = f"Observation: {results}"
        else:
            current_prompt = (
                "Observation: Please output either 'Action: search_web[<query>]' "
                "or 'Final Answer: <text>'."
            )

    return "Reached maximum iteration limit without finalizing an answer."

if __name__ == "__main__":
    print("\n=== Web Search Q&A Agent Ready ===")
    print("Type your question and press Enter. Type 'exit' or 'quit' to stop.\n")
    
    while True:
        try:
            user_input = input("\nYou: ").strip()
            
            if not user_input:
                continue
            
            if user_input.lower() in ["exit", "quit", "q"]:
                print("Exiting agent. Goodbye!")
                break
                
            answer = ask_agent(user_input)
            print("\nFinal Output:\n", answer)
            print("-" * 50)
            
        except KeyboardInterrupt:
            print("\nSession interrupted. Exiting.")
            break
