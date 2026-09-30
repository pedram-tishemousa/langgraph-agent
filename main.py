import sqlite3
from typing import Annotated, Sequence, Literal
from typing_extensions import TypedDict
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, START, END
from langchain_ollama import ChatOllama
from langchain_core.tools import tool
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver

# ۱. کتابخانه‌های مورد نیاز برای RAG
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]

# ۲. راه‌اندازی دیتابیس برداری به صورت سراسری (تا با هر بار اجرای ابزار از نو لود نشود)
embeddings = OllamaEmbeddings(model="nomic-embed-text")
vectorstore = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
# جستجوگر تنظیم می‌شود تا فقط ۱ نتیجه (مرتبط‌ترین بخش داکیومنت) را برگرداند
retriever = vectorstore.as_retriever(search_kwargs={"k": 1})

# ابزار اول: جستجو در دیتابیس
@tool
def check_server_db(server_name: str) -> str:
    """Query the database to get the real-time status of a server."""
    conn = sqlite3.connect('company.db')
    cursor = conn.cursor()
    cursor.execute("SELECT ip_address, status, cpu_load FROM servers WHERE name = ?", (server_name.lower(),))
    result = cursor.fetchone()
    conn.close()
    if result:
        ip, status, cpu = result
        return f"Server Found! IP: {ip}, Status: {status}, CPU Load: {cpu}"
    return "Server not found."

# ابزار دوم: خواندن فایل لاگ
@tool
def read_server_logs(server_name: str) -> str:
    """Get the latest log messages and errors for a specific server."""
    logs = {
        "web": "ERROR 502: Bad Gateway at 10:45 AM. Too many connections.",
        "database": "INFO: Backup completed successfully at 02:00 AM.",
        "cache": "WARNING: Memory usage at 95%."
    }
    return logs.get(server_name.lower(), "هیچ لاگی یافت نشد.")

# ۳. ابزار سوم: جستجو در مستندات برای یافتن راهکار
@tool
def search_knowledge_base(query: str) -> str:
    """Search company documentation to find solutions for errors, server operations, or guidelines."""
    docs = retriever.invoke(query)
    if docs:
        return docs[0].page_content
    return "هیچ راهنمایی در مستندات برای این مشکل یافت نشد."

# ۴. اضافه کردن ابزار سوم به لیست
tools = [check_server_db, read_server_logs, search_knowledge_base]

model = ChatOllama(
    model="llama3.1",
    temperature=0
).bind_tools(tools)

def call_model(state: AgentState):
    messages = state["messages"]
    
    # ساخت یک لیست برای نگهداری تمام سرورهای بررسی‌شده به ترتیب زمان
    discussed_servers = []
    
    # خواندن پیام‌ها از ابتدا به انتها برای حفظ ترتیب زمانی (Chronological Order)
    for msg in messages:
        if hasattr(msg, 'tool_calls') and msg.tool_calls:
            for tool in msg.tool_calls:
                if 'server_name' in tool.get('args', {}):
                    srv = tool['args']['server_name']
                    # اگر سرور قبلاً در لیست نیست، آن را اضافه می‌کنیم
                    if srv not in discussed_servers:
                        discussed_servers.append(srv)

    # قالب‌بندی لیست برای تزریق به پرامپت
    if discussed_servers:
        servers_context = f"Servers discussed so far (in order from first to last): {', '.join(discussed_servers)}"
    else:
        servers_context = "No servers discussed yet."

    # تزریق تاریخچه کامل به پرامپت سیستم
    system_prompt = SystemMessage(
        content=f"""You are an expert IT Assistant. You execute backend tools yourself.
        
[CONVERSATION CONTEXT]
{servers_context}

Tool Routing Rules:
1. To check IP, status, or load -> call 'check_server_db'.
2. To check logs or errors -> call 'read_server_logs'.
3. To find a solution or fix an error -> call 'search_knowledge_base'.
4. For references to the past (e.g., "first server", "last server"): Read the [CONVERSATION CONTEXT] array above and answer directly. DO NOT call tools.

CRITICAL RESTRICTIONS:
- DO NOT invent solutions.
- If checking logs, ONLY report the log content.
- When asked for a solution, use 'search_knowledge_base'.
- Always respond in fluent Persian."""
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

memory = MemorySaver()
app = workflow.compile(checkpointer=memory)

if __name__ == "__main__":
    print("🤖 ایجنت مجهز به RAG بیدار شد!")
    config = {"configurable": {"thread_id": "rag_session_1"}}
    while True:
        user_input = input("\nشما: ")
        if user_input.lower() in ["exit", "خروج"]:
            break
        inputs = {"messages": [HumanMessage(content=user_input)]}
        for chunk in app.stream(inputs, config=config, stream_mode="values"):
            last_msg = chunk["messages"][-1]
            if last_msg.type == "ai" and last_msg.tool_calls:
                print(f"⚙️ [ایجنت در حال استفاده از ابزار: {last_msg.tool_calls[0]['name']}]")
            elif last_msg.type == "ai" and not last_msg.tool_calls:
                print(f"ایجنت: {last_msg.content}")