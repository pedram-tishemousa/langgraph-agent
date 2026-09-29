import sqlite3
from typing import Annotated, Sequence, Literal
from typing_extensions import TypedDict
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, START, END
from langchain_ollama import ChatOllama
from langchain_core.tools import tool
from langgraph.prebuilt import ToolNode

# ۱. وارد کردن ماژول حافظه محلی
from langgraph.checkpoint.memory import MemorySaver 

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]

@tool
def check_server_db(server_name: str) -> str:
    """
    Query the database to get the real-time status of a server.
    Args:
        server_name (str): The exact name of the server to check. Examples: 'web', 'database', 'cache'.
    """
    conn = sqlite3.connect('company.db')
    cursor = conn.cursor()
    cursor.execute("SELECT ip_address, status, cpu_load FROM servers WHERE name = ?", (server_name.lower(),))
    result = cursor.fetchone()
    conn.close()
    
    if result:
        ip, status, cpu = result
        return f"Server Found! IP: {ip}, Status: {status}, CPU Load: {cpu}"
    else:
        return "Server not found in the database."

tools = [check_server_db]

model = ChatOllama(
    model="qwen2.5:1.5b",
    temperature=0
).bind_tools(tools)

def call_model(state: AgentState):
    messages = state["messages"]
    system_prompt = SystemMessage(
        content="""You are an IT assistant connected to a live database.
        You MUST ALWAYS use the 'check_server_db' tool to fetch information if the user asks about a server.
        CRITICAL: Extract the server name from the user's message (e.g. "web", "database", "cache") and pass it to the tool.
        DO NOT guess the IP or status. Use the database!"""
    )
    response = model.invoke([system_prompt] + list(messages))
    return {"messages": [response]}

def should_continue(state: AgentState) -> Literal["tools", END]:
    messages = state["messages"]
    last_message = messages[-1]
    if last_message.tool_calls:
        return "tools"
    return END

workflow = StateGraph(AgentState)
workflow.add_node("agent", call_model)
workflow.add_node("tools", ToolNode(tools))

workflow.add_edge(START, "agent")
workflow.add_conditional_edges("agent", should_continue)
workflow.add_edge("tools", "agent")

# ۲. ساخت یک شیء حافظه و اتصال آن به گراف در زمان کامپایل
memory = MemorySaver()
app = workflow.compile(checkpointer=memory)

if __name__ == "__main__":
    print("🤖 ایجنت متصل به دیتابیس (با حافظه فعال) بیدار شد!")
    
    # ۳. تعریف یک شناسه یکتا (Thread ID) برای این مکالمه
    config = {"configurable": {"thread_id": "session_1"}}
    
    while True:
        user_input = input("\nشما: ")
        if user_input.lower() in ["exit", "خروج"]:
            break
            
        inputs = {"messages": [HumanMessage(content=user_input)]}
        
        # ۴. پاس دادن تنظیمات (شامل شناسه مکالمه) به جریان اجرای گراف
        for chunk in app.stream(inputs, config=config, stream_mode="values"):
            last_msg = chunk["messages"][-1]
            
            if last_msg.type == "ai" and last_msg.tool_calls:
                tool_name = last_msg.tool_calls[0]['name']
                arg_value = last_msg.tool_calls[0]['args'].get('server_name', 'نامشخص')
                print(f"🔍 [ایجنت در حال جستجوی '{arg_value}'...]")
            elif last_msg.type == "ai" and not last_msg.tool_calls:
                print(f"ایجنت: {last_msg.content}")