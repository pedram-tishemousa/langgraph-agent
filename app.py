import streamlit as st
from langchain_core.messages import HumanMessage
from main import app  # گراف ایجنت را از فایل قبلی وارد می‌کنیم

# تنظیمات ظاهری صفحه
st.set_page_config(page_title="AI Server Monitor", page_icon="🖥️")
st.title("دستیار هوشمند مانیتورینگ سرور 📊")

# ۱. راه‌اندازی حافظه رابط کاربری
# استریم‌لیت با هر کلیک کاربر کل کد را از اول اجرا می‌کند، پس باید تاریخچه را در session_state ذخیره کنیم
if "messages" not in st.session_state:
    st.session_state.messages = []
    
# یک شناسه یکتا برای اتصال حافظه گراف به این سشن وب
if "thread_id" not in st.session_state:
    st.session_state.thread_id = "web_session_1"

# ۲. نمایش پیام‌های قبلی در صفحه
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ۳. دریافت پیام جدید از کاربر
if prompt := st.chat_input("سوال خود را درباره وضعیت سرورها بپرسید... (مثلا: سرور cache چطوره؟)"):
    
    # چاپ پیام کاربر در صفحه و ذخیره در تاریخچه UI
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # ۴. ارسال پیام به مغز ایجنت (LangGraph)
    config = {"configurable": {"thread_id": st.session_state.thread_id}}
    inputs = {"messages": [HumanMessage(content=prompt)]}
    
    # نمایش انیمیشن لودینگ تا زمانی که ایجنت فکر می‌کند و در دیتابیس می‌گردد
    with st.spinner("ایجنت در حال بررسی..."):
        final_answer = ""
        
        # اجرای گراف و گرفتن آخرین جواب متنی
        for chunk in app.stream(inputs, config=config, stream_mode="values"):
            last_msg = chunk["messages"][-1]
            
            # فقط پیام‌های نهایی (غیر از درخواست‌های ابزار) را می‌گیریم
            if last_msg.type == "ai" and not last_msg.tool_calls:
                final_answer = last_msg.content

    # ۵. چاپ جواب ایجنت در صفحه و ذخیره در تاریخچه UI
    with st.chat_message("assistant"):
        st.markdown(final_answer)
    st.session_state.messages.append({"role": "assistant", "content": final_answer})