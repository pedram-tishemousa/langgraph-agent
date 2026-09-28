from typing import Annotated, Sequence, Literal
from typing_extensions import TypedDict
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, START, END
from langchain_ollama import ChatOllama
from langchain_core.tools import tool
from langgraph.prebuilt import ToolNode

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]

# ۱. تعریف ابزار با دکوراتور @tool (توضیحات تابع برای فهم مدل بسیار مهم است)
@tool
def get_server_status(server_name: str) -> str:
    """
    Get the current status of a specific server.
    Args:
        server_name (str): The exact name of the server to check. Examples: 'web', 'database'.
    """
    statuses = {
        "database": "سرور دیتابیس آنلاین است و پینگ آن 12ms است.",
        "web": "سرور وب آفلاین است (خطای 502)."
    }
    return statuses.get(server_name.lower(), "سروری با این نام یافت نشد.")

# لیست ابزارهای در دسترس ایجنت
tools = [get_server_status]

# ۲. معرفی مدل و Bind کردن ابزارها به آن
model = ChatOllama(
    model="qwen2.5:1.5b",
    temperature=0
).bind_tools(tools)

def call_model(state: AgentState):
    messages = state["messages"]
    system_prompt = SystemMessage(
        content="""You are an AI tool-calling assistant. 
        You MUST use the 'get_server_status' tool to answer.
        CRITICAL INSTRUCTION: Extract the server name directly from the user's prompt. 
        If the user says "web", pass "web" as the server_name parameter. 
        If the user says "database", pass "database" as the server_name parameter.
        DO NOT ask the user for the server name. Extract it yourself!"""
    )
    response = model.invoke(messages)
    return {"messages": [response]}

# ۳. تابع مسیریاب (لبه شرطی)
def should_continue(state: AgentState) -> Literal["tools", END]:
    messages = state["messages"]
    last_message = messages[-1]
    # اگر مدل تصمیم گرفته ابزاری را صدا بزند، مسیر را به نود tools بفرست
    if last_message.tool_calls:
        return "tools"
    # در غیر این صورت کار تمام است
    return END

# ۴. ساخت گراف
workflow = StateGraph(AgentState)

# اضافه کردن نود مدل و نود از پیش ساخته‌شده‌ی ابزارها
workflow.add_node("agent", call_model)
workflow.add_node("tools", ToolNode(tools))

# ۵. اتصال نودها و رسم مسیر جریان
workflow.add_edge(START, "agent")
workflow.add_conditional_edges("agent", should_continue)
workflow.add_edge("tools", "agent") # بعد از اجرای ابزار، نتیجه به مدل برمی‌گردد

app = workflow.compile()

if __name__ == "__main__":
    print("🤖 ایجنت هوشمند بیدار شد! (برای خروج 'exit' را تایپ کنید)")
    while True:
        user_input = input("\nشما: ")
        if user_input.lower() in ["exit", "خروج"]:
            break
            
        inputs = {"messages": [HumanMessage(content=user_input)]}
        for chunk in app.stream(inputs, stream_mode="values"):
            last_msg = chunk["messages"][-1]
            
            # چاپ لاگ برای درک بهتر جریان کار ایجنت
            if last_msg.type == "ai" and last_msg.tool_calls:
                print(f"⚙️ [ایجنت در حال استفاده از ابزار {last_msg.tool_calls[0]['name']} است...]")
            elif last_msg.type == "ai" and not last_msg.tool_calls:
                print(f"ایجنت: {last_msg.content}")