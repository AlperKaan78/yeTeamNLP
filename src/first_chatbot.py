from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

# --- Define tools ---
@tool
def search_web(query: str) -> str:
    """
    Searches the web using DuckDuckGo and returns the top 3 text results.
    """
    from ddgs import DDGS
    results = []
    with DDGS() as ddgs:
        for r in ddgs.text(query, max_results=3):
            results.append(r["body"])
    return "\n".join(results)

@tool
def analyze_data(data: str) -> str:
    """Analyze given data and return insights."""
    return f"Analyzed data: {data}"

@tool
def send_email(recipient: str, message: str) -> str:
    """Pretend to send an email."""
    return f"Email sent to {recipient} with message: {message}"

llm = ChatGoogleGenerativeAI(model="gemini-2.5-pro")

# --- Create agent ---
agent = create_agent(
    model=llm,   # ← Use OpenAI model
    tools=[search_web, analyze_data, send_email],
    system_prompt="You are a concise AI assistant. "
    "Always give short, direct answers and prefer comparisons when explaining concepts."
)

def get_result_text(result):
    all_res = result["messages"][-1].content
    res = all_res[0]
    return res["text"]

while True: 
    user_content = input(">>")
    result = agent.invoke({
    "messages": [
        {"role": "user", "content": user_content}
    ]
    })
    print(get_result_text(result))
    if user_content == "exit":
        break
