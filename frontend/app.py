from __future__ import annotations

import base64
import os
import uuid
from pathlib import Path

import streamlit as st

from api_client import APIError, LawChatAPI
from components import (
    STATUS_LABELS,
    answer_for_display,
    render_citations,
    render_message,
)


st.set_page_config(
    page_title="LawChat",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE_DIR = Path(__file__).resolve().parent
PUBLIC_BASE_URL = os.getenv("LAWCHAT_PUBLIC_BASE_URL", "http://localhost:8501")


@st.cache_resource
def api_client() -> LawChatAPI:
    return LawChatAPI()


api = api_client()


def _background_css() -> str:
    image_path = BASE_DIR / "assets" / "lawchat-bg.jpg"
    if not image_path.exists():
        return ""
    encoded = base64.b64encode(image_path.read_bytes()).decode()
    return f"""
    .stApp::before {{
        content: ""; position: fixed; inset: 0;
        background: url("data:image/png;base64,{encoded}") center/cover no-repeat;
        opacity: .58; filter: saturate(1.18) contrast(1.12) brightness(1.08); pointer-events: none;
    }}
    .stApp::after {{
        content: ""; position: fixed; inset: 0;
        background:
            radial-gradient(circle at 52% 14%, rgba(255,205,95,.14), transparent 42%),
            linear-gradient(180deg, rgba(12,13,17,.08), rgba(9,10,14,.52));
        pointer-events: none;
    }}
    """


st.markdown(
    f"""
    <style>
    html, body, .stApp {{ background:#101116; color:#f7f7f8; }}
    {_background_css()}
    [data-testid="stAppViewContainer"] > .main,
    [data-testid="stSidebar"] {{ position:relative; z-index:2; }}
    [data-testid="stSidebar"] {{
        background:rgba(11,12,16,.97); border-right:1px solid rgba(255,255,255,.07);
    }}
    .block-container {{ max-width:980px; padding-top:1.3rem; padding-bottom:7rem; }}
    div[data-testid="stChatMessage"] {{
        padding:15px 17px; margin-bottom:12px; border-radius:18px;
        border:1px solid rgba(255,255,255,.07); background:rgba(23,25,31,.84);
    }}
    div[data-testid="stChatInput"] > div {{
        border-radius:24px; background:rgba(30,32,39,.97);
    }}
    .lawchat-brand {{ font-size:21px; font-weight:700; margin-bottom:4px; }}
    .lawchat-subtitle {{ color:#999faa; font-size:12px; margin-bottom:24px; }}
    .hero {{ font-size:43px; line-height:1.1; font-weight:680; margin:65px 0 16px; }}
    .hero-note {{ color:#a4a8b1; max-width:620px; line-height:1.65; }}
    .unofficial {{
        color:#d8bd8a; border:1px solid rgba(216,189,138,.25);
        background:rgba(216,189,138,.08); padding:7px 10px; border-radius:10px;
        font-size:11px; margin-bottom:18px;
    }}
    #MainMenu, footer {{ visibility:hidden; }}
    </style>
    """,
    unsafe_allow_html=True,
)


def fail(message: str) -> None:
    st.error(message)


def show_shared(token: str) -> None:
    try:
        payload = api.get_shared_conversation(token)
    except APIError as exc:
        fail(str(exc))
        st.stop()
    conversation = payload["conversation"]
    st.markdown('<div class="lawchat-brand">⚖️ LawChat</div>', unsafe_allow_html=True)
    st.caption("Cuộc trò chuyện được chia sẻ · Chỉ đọc")
    st.title(conversation["title"])
    for message in payload["messages"]:
        render_message(message)
    st.stop()


share_token = st.query_params.get("share")
if share_token:
    show_shared(str(share_token))


if "active_conversation_id" not in st.session_state:
    st.session_state.active_conversation_id = None
if "active_project_id" not in st.session_state:
    st.session_state.active_project_id = None

try:
    projects = api.list_projects()
    conversations = api.list_conversations()
except APIError as exc:
    fail(str(exc))
    st.info("Hãy kiểm tra FastAPI, migration và LAWCHAT_API_BASE_URL.")
    st.stop()

project_names = {item["id"]: item["name"] for item in projects}


with st.sidebar:
    st.markdown('<div class="lawchat-brand">⚖️ LawChat</div>', unsafe_allow_html=True)
    st.caption("Vietnamese Legal RAG")
    if st.button("＋ Cuộc trò chuyện mới", use_container_width=True):
        st.session_state.active_conversation_id = None
        st.rerun()

    with st.expander("Tạo dự án"):
        with st.form("create_project", clear_on_submit=True):
            project_name = st.text_input("Tên dự án")
            if st.form_submit_button("Tạo", use_container_width=True):
                if project_name.strip():
                    try:
                        created = api.create_project(project_name.strip())
                        st.session_state.active_project_id = created["id"]
                        st.rerun()
                    except APIError as exc:
                        fail(str(exc))

    project_options = {"Tất cả cuộc trò chuyện": None}
    project_options.update({item["name"]: item["id"] for item in projects})
    current_project_label = next(
        (label for label, value in project_options.items()
         if value == st.session_state.active_project_id),
        "Tất cả cuộc trò chuyện",
    )
    selected_project_label = st.selectbox(
        "Dự án", list(project_options),
        index=list(project_options).index(current_project_label),
    )
    selected_project_id = project_options[selected_project_label]
    if selected_project_id != st.session_state.active_project_id:
        st.session_state.active_project_id = selected_project_id
        st.rerun()

    if selected_project_id:
        with st.expander("Quản lý dự án"):
            with st.form("rename_project"):
                renamed_project = st.text_input(
                    "Tên dự án", value=project_names.get(selected_project_id, "")
                )
                if st.form_submit_button("Lưu tên", use_container_width=True):
                    try:
                        api.rename_project(selected_project_id, renamed_project)
                        st.rerun()
                    except APIError as exc:
                        fail(str(exc))
            if st.button(
                "Xóa dự án", key="delete-active-project", use_container_width=True
            ):
                try:
                    api.delete_project(selected_project_id)
                    st.session_state.active_project_id = None
                    st.rerun()
                except APIError as exc:
                    fail(str(exc))

    st.caption("CUỘC TRÒ CHUYỆN")
    visible = [
        item for item in conversations
        if selected_project_id is None or item.get("project_id") == selected_project_id
    ]
    for conversation in visible:
        row = st.columns([7, 1, 1])
        label = ("📌 " if conversation["pinned"] else "") + conversation["title"]
        if row[0].button(
            label, key=f"open-{conversation['id']}", use_container_width=True
        ):
            st.session_state.active_conversation_id = conversation["id"]
            st.rerun()
        if row[1].button("★", key=f"pin-{conversation['id']}"):
            try:
                api.update_conversation(
                    conversation["id"], pinned=not conversation["pinned"]
                )
                st.rerun()
            except APIError as exc:
                fail(str(exc))
        if row[2].button("×", key=f"delete-{conversation['id']}"):
            try:
                api.delete_conversation(conversation["id"])
                if st.session_state.active_conversation_id == conversation["id"]:
                    st.session_state.active_conversation_id = None
                st.rerun()
            except APIError as exc:
                fail(str(exc))

    st.divider()
    online = api.health()
    st.caption("● API online" if online else "● API unavailable")


st.markdown('<div class="lawchat-brand">⚖️ LawChat</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="unofficial">Hệ thống hỗ trợ tra cứu thử nghiệm, không phải cổng thông tin chính thức hoặc ý kiến tư vấn pháp lý.</div>',
    unsafe_allow_html=True,
)

active_id = st.session_state.active_conversation_id
active_conversation = None
messages: list[dict] = []
if active_id:
    try:
        active_conversation = api.get_conversation(active_id)
        messages = api.list_messages(active_id)
    except APIError as exc:
        fail(str(exc))
        st.session_state.active_conversation_id = None
        active_id = None


if active_conversation:
    title_col, share_col = st.columns([8, 2])
    title_col.subheader(active_conversation["title"])
    project_name = project_names.get(active_conversation.get("project_id"))
    if project_name:
        title_col.caption(f"📁 {project_name}")
    if share_col.button("Chia sẻ", use_container_width=True):
        try:
            token = api.share_conversation(active_conversation["id"])
            st.session_state.share_url = (
                PUBLIC_BASE_URL.rstrip("/") + "/?share=" + token
            )
        except APIError as exc:
            fail(str(exc))
    if st.session_state.get("share_url"):
        st.text_input("Liên kết chỉ đọc", st.session_state.share_url)
    with st.expander("Tùy chọn hội thoại"):
        conversation_project_options = {"Không thuộc dự án": None}
        conversation_project_options.update(
            {item["name"]: item["id"] for item in projects}
        )
        current_label = next(
            (label for label, value in conversation_project_options.items()
             if value == active_conversation.get("project_id")),
            "Không thuộc dự án",
        )
        with st.form("conversation_settings"):
            new_title = st.text_input("Tên hội thoại", active_conversation["title"])
            new_project_label = st.selectbox(
                "Dự án",
                list(conversation_project_options),
                index=list(conversation_project_options).index(current_label),
            )
            if st.form_submit_button("Lưu thay đổi", use_container_width=True):
                try:
                    api.update_conversation(
                        active_conversation["id"],
                        title=new_title,
                        project_id=conversation_project_options[new_project_label],
                        update_project=True,
                    )
                    st.rerun()
                except APIError as exc:
                    fail(str(exc))
        if active_conversation.get("is_shared") and st.button(
            "Dừng chia sẻ", use_container_width=True
        ):
            try:
                api.stop_sharing_conversation(active_conversation["id"])
                st.session_state.pop("share_url", None)
                st.rerun()
            except APIError as exc:
                fail(str(exc))
    for message in messages:
        render_message(message)
else:
    st.markdown(
        '<div class="hero">Tôi có thể giúp gì<br>cho bạn hôm nay?</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="hero-note">Tra cứu pháp luật Việt Nam, đối chiếu hiệu lực và nhận câu trả lời có căn cứ được kiểm tra trước khi hiển thị.</div>',
        unsafe_allow_html=True,
    )
    suggestions = [
        "Người lao động được nghỉ phép bao nhiêu ngày?",
        "Điều kiện chuyển nhượng quyền sử dụng đất là gì?",
        "Vượt đèn đỏ bị xử phạt như thế nào?",
        "Khi nào hợp đồng lao động bị vô hiệu?",
    ]
    cols = st.columns(2)
    for index, suggestion in enumerate(suggestions):
        if cols[index % 2].button(suggestion, use_container_width=True):
            st.session_state.pending_prompt = suggestion
            st.rerun()


prompt = st.chat_input("Hỏi LawChat về pháp luật Việt Nam…")
if st.session_state.get("pending_prompt"):
    prompt = st.session_state.pop("pending_prompt")

if prompt and prompt.strip():
    prompt = prompt.strip()
    try:
        if not active_id:
            conversation = api.create_conversation(
                prompt[:80], st.session_state.active_project_id
            )
            active_id = conversation["id"]
            st.session_state.active_conversation_id = active_id

        with st.chat_message("user"):
            st.markdown(prompt)

        final_result = None
        stream_failed = False
        with st.chat_message("assistant"):
            status_box = st.status("Đang tiếp nhận yêu cầu…", expanded=True)
            answer_placeholder = st.empty()
            streamed_answer = ""
            try:
                for event_name, event_data in api.stream_chat(
                    conversation_id=active_id,
                    client_message_id=uuid.uuid4().hex,
                    message=prompt,
                ):
                    if event_name in STATUS_LABELS:
                        status_box.update(label=STATUS_LABELS[event_name])
                    elif event_name == "answer.delta":
                        streamed_answer += str(event_data.get("text", ""))
                        answer_placeholder.markdown(
                            answer_for_display(streamed_answer) + "▌"
                        )
                    elif event_name == "chat.completed":
                        final_result = event_data.get("result") or {}
                    elif event_name == "chat.failed":
                        stream_failed = True
                        status_box.update(label="Không thể hoàn tất", state="error")
                        fail(str(event_data.get("message", "Yêu cầu thất bại")))
                if final_result is not None:
                    answer_placeholder.markdown(
                        answer_for_display(
                            final_result.get("answer") or streamed_answer
                        )
                    )
                    status_box.update(label="Hoàn tất", state="complete", expanded=False)
                    render_citations(final_result.get("citations") or [])
                elif not stream_failed:
                    status_box.update(label="Stream kết thúc không đầy đủ", state="error")
            except APIError as exc:
                stream_failed = True
                status_box.update(label="Mất kết nối", state="error")
                fail(str(exc))
        if not stream_failed:
            st.rerun()
    except APIError as exc:
        fail(str(exc))

st.caption(
    "LawChat có thể tạo ra thông tin chưa chính xác. Hãy kiểm tra văn bản nguồn trước khi áp dụng."
)
