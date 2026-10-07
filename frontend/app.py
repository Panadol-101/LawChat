from __future__ import annotations

import base64
import os
import uuid
from pathlib import Path

import streamlit as st

from api_client import REGISTER_SUCCESS_MESSAGE, APIError, LawChatAPI
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



def get_api_client() -> LawChatAPI:
    if "api_client" not in st.session_state:
        st.session_state.api_client = LawChatAPI()

    return st.session_state.api_client


api = get_api_client()

def login_screen() -> None:
    st.markdown(
        '<div class="lawchat-brand">⚖ LawChat</div>',
        unsafe_allow_html=True,
    )

    login_tab, register_tab = st.tabs(["Đăng nhập", "Đăng ký"])

    with login_tab:
        st.subheader("Đăng nhập")

        with st.form("login_form"):
            username = st.text_input(
                "Tên đăng nhập",
                key="login_username",
            )
            password = st.text_input(
                "Mật khẩu",
                type="password",
                key="login_password",
            )

            if st.form_submit_button(
                "Đăng nhập",
                use_container_width=True,
            ):
                if not username.strip() or not password:
                    st.warning(
                        "Vui lòng nhập đầy đủ tên đăng nhập và mật khẩu."
                    )
                else:
                    try:
                        user = api.login(
                            username.strip(),
                            password,
                        )

                        if user.get("requires_totp"):
                            # No session is issued until the TOTP step passes.
                            st.warning(
                                "Tài khoản quản trị phải đăng nhập bằng mã TOTP "
                                "tại trang quản trị."
                            )
                            st.stop()

                        st.session_state.authenticated = True
                        st.session_state.auth_user = user

                        st.rerun()

                    except APIError as exc:
                        fail(str(exc))

    with register_tab:
        st.subheader("Tạo tài khoản")

        with st.form("register_form"):
            username = st.text_input(
                "Tên đăng nhập",
                key="register_username",
            )
            password = st.text_input(
                "Mật khẩu",
                type="password",
                key="register_password",
            )
            confirm_password = st.text_input(
                "Nhập lại mật khẩu",
                type="password",
                key="register_confirm_password",
            )

            if st.form_submit_button(
                "Đăng ký",
                use_container_width=True,
            ):
                if not username.strip():
                    st.warning("Vui lòng nhập tên đăng nhập.")
                elif not 3 <= len(username.strip()) <= 50 or not all(
                    char.isascii() and (char.isalnum() or char == "_")
                    for char in username.strip()
                ):
                    st.warning(
                        "Tên đăng nhập phải có từ 3 đến 50 ký tự, chỉ gồm chữ cái "
                        "không dấu, chữ số hoặc dấu gạch dưới (_)."
                    )
                elif not 8 <= len(password) <= 128:
                    st.warning("Mật khẩu phải có từ 8 đến 128 ký tự.")
                elif password != confirm_password:
                    st.warning("Mật khẩu nhập lại không khớp.")
                else:
                    try:
                        api.register(
                            username.strip(),
                            password,
                            confirm_password,
                        )

                        st.success(REGISTER_SUCCESS_MESSAGE)

                    except APIError as exc:
                        fail(str(exc))


def require_auth() -> None:
    if st.session_state.get("authenticated"):
        return

    try:
        user = api.me()
        st.session_state.authenticated = True
        st.session_state.auth_user = user
        return
    except APIError:
        pass


    login_screen()
    st.stop()

def _background_css() -> str:
    image_path = BASE_DIR / "assets" / "lawchat-bg.jpg"
    if not image_path.exists():
        return ""

    encoded = base64.b64encode(image_path.read_bytes()).decode()

    return f"""
    .stApp::before {{
        content: "";
        position: fixed;
        inset: 0;
        z-index: 0;
        background: url("data:image/jpeg;base64,{encoded}") center/cover no-repeat;
        opacity: .14;
        filter: saturate(1.05) contrast(1.02) brightness(1.05);
        pointer-events: none;
    }}

    .stApp::after {{
        content: "";
        position: fixed;
        inset: 0;
        z-index: 1;
        background: rgba(10, 11, 15, .18);
        pointer-events: none;
    }}
    """


st.markdown(
    f"""
    <style>

    /* =========================
       Streamlit header / toolbar
       ========================= */

    header[data-testid="stHeader"] {{
        display: none !important;
    }}

    div[data-testid="stDecoration"] {{
        display: none !important;
    }}

    div[data-testid="stToolbar"] {{
        display: none !important;
    }}

    footer {{
        display: none !important;
    }}


    /* =========================
       App
       ========================= */

    html, body, .stApp {{
        background:#101116;
        color:#f7f7f8;
    }}

    {_background_css()}

    [data-testid="stAppViewContainer"] {{
        padding-top: 0 !important;
    }}

    [data-testid="stAppViewContainer"] > .main,
    [data-testid="stSidebar"] {{
        position: relative;
        z-index: 2;
    }}

    [data-testid="stSidebar"] {{
        background:rgba(11,12,16,.97);
        border-right:1px solid rgba(255,255,255,.07);
    }}

    .block-container {{
        max-width:980px;
        padding-top:1rem !important;
        padding-bottom:7rem;
    }}


    /* =========================
       Chat messages
       ========================= */

    div[data-testid="stChatMessage"] {{
        padding:15px 17px;
        margin-bottom:12px;
        border-radius:18px;
        border:1px solid rgba(255,255,255,.07);
        background:rgba(23,25,31,.84);
    }}


    /* =========================
       Chat input - remove black bar
       ========================= */

    div[data-testid="stBottom"] {{
        background: transparent !important;
    }}

    div[data-testid="stBottomBlockContainer"] {{
        background: transparent !important;
    }}

    div[data-testid="stBottomBlockContainer"] > div {{
        background: transparent !important;
    }}

    div[data-testid="stChatInput"] {{
        background: transparent !important;
    }}

    div[data-testid="stChatInput"] > div {{
        border-radius:24px;
        background:rgba(25,27,34,.98);
        border:1px solid rgba(255,255,255,.10);
    }}


    /* =========================
       LawChat
       ========================= */

    .lawchat-brand {{
        font-size:21px;
        font-weight:700;
        margin-bottom:4px;
    }}

    .lawchat-subtitle {{
        color:#999faa;
        font-size:12px;
        margin-bottom:24px;
    }}

    .hero {{
        font-size:43px;
        line-height:1.1;
        font-weight:680;
        margin:65px 0 16px;
    }}

    .hero-note {{
        color:#a4a8b1;
        max-width:620px;
        line-height:1.65;
    }}

    .unofficial {{
        color:#d8bd8a;
        border:1px solid rgba(216,189,138,.25);
        background:rgba(216,189,138,.08);
        padding:7px 10px;
        border-radius:10px;
        font-size:11px;
        margin-bottom:18px;
    }}

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

require_auth()

try:
    projects = api.list_projects()
    conversations = api.list_conversations()
except APIError as exc:
    fail(str(exc))
    st.info("Hãy kiểm tra FastAPI, migration và LAWCHAT_API_BASE_URL.")
    st.stop()

project_names = {item["id"]: item["name"] for item in projects}


with st.sidebar:

    # ============================================================
    # LAWCHAT BRAND
    # ============================================================

    st.markdown(
        '<div class="lawchat-brand">⚖️ LawChat</div>',
        unsafe_allow_html=True,
    )

    st.caption("Trợ lý pháp luật Việt Nam")


    # ============================================================
    # NEW CHAT
    # ============================================================

    if st.button(
        "",
        icon=":material/edit_square:",
        help="Cuộc trò chuyện mới",
        type="secondary",
        use_container_width=True,
        key="new-chat",
    ):
        st.session_state.active_conversation_id = None
        st.session_state.pop("share_url", None)
        st.rerun()


    # ============================================================
    # CREATE PROJECT
    # ============================================================

    with st.expander("Tạo dự án"):

        with st.form(
            "create_project",
            clear_on_submit=True,
        ):

            project_name = st.text_input(
                "Tên dự án"
            )

            if st.form_submit_button(
                "Tạo",
                use_container_width=True,
            ):

                if project_name.strip():

                    try:

                        created = api.create_project(
                            project_name.strip()
                        )

                        st.session_state.active_project_id = (
                            created["id"]
                        )

                        st.session_state.active_conversation_id = None

                        st.rerun()

                    except APIError as exc:
                        fail(str(exc))


    # ============================================================
    # PROJECT SELECTOR
    # ============================================================

    project_options = {
        "Tất cả cuộc trò chuyện": None
    }

    project_options.update(
        {
            item["name"]: item["id"]
            for item in projects
        }
    )


    current_project_label = next(
        (
            label
            for label, value in project_options.items()
            if value == st.session_state.active_project_id
        ),
        "Tất cả cuộc trò chuyện",
    )


    selected_project_label = st.selectbox(
        "Dự án",
        list(project_options),
        index=list(project_options).index(
            current_project_label
        ),
        key="project-selector",
    )


    selected_project_id = project_options[
        selected_project_label
    ]


    if selected_project_id != st.session_state.active_project_id:

        st.session_state.active_project_id = (
            selected_project_id
        )

        st.session_state.active_conversation_id = None
        st.session_state.pop("share_url", None)

        st.rerun()


    # ============================================================
    # PROJECT ACTIONS
    # ============================================================

    if selected_project_id:

        project_menu_col, project_chat_col = st.columns(
            [1, 1],
            gap="small",
        )


        # ========================================================
        # PROJECT "..."
        # ========================================================

        with project_menu_col:

            with st.popover(
                "",
                icon=":material/more_horiz:",
                help="Tùy chọn dự án",
                use_container_width=True,
            ):

                st.caption(
                    project_names.get(
                        selected_project_id,
                        "Dự án",
                    )
                )


                # ------------------------------------------------
                # PROJECT HOME
                # ------------------------------------------------

                if st.button(
                    "Trang chủ dự án",
                    icon=":material/home:",
                    use_container_width=True,
                    key=f"project-home-{selected_project_id}",
                ):

                    st.session_state.active_conversation_id = None
                    st.session_state.pop("share_url", None)

                    st.rerun()


                # ------------------------------------------------
                # RENAME PROJECT
                # ------------------------------------------------

                with st.popover(
                    "Đổi tên dự án",
                    icon=":material/edit:",
                    use_container_width=True,
                ):

                    renamed_project = st.text_input(
                        "Tên dự án",
                        value=project_names.get(
                            selected_project_id,
                            "",
                        ),
                        key=f"rename-project-input-{selected_project_id}",
                    )


                    if st.button(
                        "Lưu",
                        type="primary",
                        use_container_width=True,
                        key=f"rename-project-save-{selected_project_id}",
                    ):

                        new_name = renamed_project.strip()

                        if not new_name:

                            st.warning(
                                "Tên dự án không được để trống."
                            )

                        else:

                            try:

                                api.rename_project(
                                    selected_project_id,
                                    new_name,
                                )

                                st.rerun()

                            except APIError as exc:
                                fail(str(exc))


                # ------------------------------------------------
                # SHARE PROJECT
                # ------------------------------------------------

                if st.button(
                    "Chia sẻ dự án",
                    icon=":material/share:",
                    use_container_width=True,
                    key=f"share-project-{selected_project_id}",
                ):

                    st.info(
                        "Chia sẻ dự án chưa được backend hỗ trợ. "
                        "Hiện tại LawChat chỉ hỗ trợ chia sẻ từng cuộc trò chuyện."
                    )


                # ------------------------------------------------
                # PIN PROJECT
                # ------------------------------------------------

                pinned_projects = st.session_state.setdefault(
                    "pinned_projects",
                    set(),
                )

                is_project_pinned = (
                    selected_project_id
                    in pinned_projects
                )


                if st.button(
                    "Bỏ ghim dự án"
                    if is_project_pinned
                    else "Ghim dự án",
                    icon=":material/push_pin:",
                    use_container_width=True,
                    key=f"pin-project-{selected_project_id}",
                ):

                    if is_project_pinned:

                        pinned_projects.remove(
                            selected_project_id
                        )

                    else:

                        pinned_projects.add(
                            selected_project_id
                        )

                    st.rerun()


                st.divider()


                # ------------------------------------------------
                # DELETE PROJECT
                # ------------------------------------------------

                if st.button(
                    "Xóa dự án",
                    icon=":material/delete:",
                    type="secondary",
                    use_container_width=True,
                    key=f"delete-project-{selected_project_id}",
                ):

                    try:

                        api.delete_project(
                            selected_project_id
                        )

                        st.session_state.active_project_id = None
                        st.session_state.active_conversation_id = None

                        pinned_projects.discard(
                            selected_project_id
                        )

                        st.session_state.pop(
                            "share_url",
                            None,
                        )

                        st.rerun()

                    except APIError as exc:
                        fail(str(exc))


        # ========================================================
        # NEW CHAT IN PROJECT
        # ========================================================

        with project_chat_col:

            if st.button(
                "",
                icon=":material/edit_square:",
                help="Cuộc trò chuyện mới trong dự án",
                type="secondary",
                use_container_width=True,
                key=f"new-project-chat-{selected_project_id}",
            ):

                st.session_state.active_conversation_id = None
                st.session_state.pop("share_url", None)

                st.rerun()


    # ============================================================
    # CONVERSATIONS
    # ============================================================

    st.markdown(
        """
        <div style="
            margin-top:18px;
            margin-bottom:8px;
            color:#999faa;
            font-size:13px;
            font-weight:600;
        ">
            CUỘC TRÒ CHUYỆN
        </div>
        """,
        unsafe_allow_html=True,
    )


    visible = [
        item
        for item in conversations
        if (
            selected_project_id is None
            or item.get("project_id") == selected_project_id
        )
    ]


    # ============================================================
    # CONVERSATION ROW
    # ============================================================

    for conversation in visible:

        conversation_id = conversation["id"]
        is_pinned = bool(
            conversation.get("pinned", False)
        )


        row = st.columns(
            [8, 1, 1],
            gap="small",
        )


        # --------------------------------------------------------
        # CHAT TITLE
        # --------------------------------------------------------

        with row[0]:

            label = conversation["title"]


            if st.button(
                label,
                key=f"open-{conversation_id}",
                use_container_width=True,
                type=(
                    "primary"
                    if (
                        st.session_state.active_conversation_id
                        == conversation_id
                    )
                    else "secondary"
                ),
            ):

                st.session_state.active_conversation_id = (
                    conversation_id
                )

                st.session_state.pop(
                    "share_url",
                    None,
                )

                st.rerun()


        # --------------------------------------------------------
        # PIN
        # --------------------------------------------------------

        with row[1]:

            if st.button(
                "",
                icon=":material/push_pin:",
                help=(
                    "Bỏ ghim"
                    if is_pinned
                    else "Ghim cuộc trò chuyện"
                ),
                type="tertiary",
                use_container_width=True,
                key=f"pin-{conversation_id}",
            ):

                try:

                    api.update_conversation(
                        conversation_id,
                        pinned=not is_pinned,
                    )

                    st.rerun()

                except APIError as exc:
                    fail(str(exc))


        # --------------------------------------------------------
        # MORE MENU
        # --------------------------------------------------------

        with row[2]:

            with st.popover(
                "",
                icon=":material/more_horiz:",
                help="Tùy chọn cuộc trò chuyện",
                use_container_width=True,
            ):

                # =================================================
                # RENAME
                # =================================================

                with st.popover(
                    "Đổi tên",
                    icon=":material/edit:",
                    use_container_width=True,
                ):

                    new_title = st.text_input(
                        "Tên cuộc trò chuyện",
                        value=conversation["title"],
                        key=f"rename-chat-input-{conversation_id}",
                    )


                    if st.button(
                        "Lưu",
                        type="primary",
                        use_container_width=True,
                        key=f"rename-chat-save-{conversation_id}",
                    ):

                        title = new_title.strip()

                        if not title:

                            st.warning(
                                "Tên cuộc trò chuyện không được để trống."
                            )

                        else:

                            try:

                                api.update_conversation(
                                    conversation_id,
                                    title=title,
                                )

                                st.rerun()

                            except APIError as exc:
                                fail(str(exc))


                # =================================================
                # PIN / UNPIN
                # =================================================

                if st.button(
                    "Bỏ ghim"
                    if is_pinned
                    else "Ghim",
                    icon=":material/push_pin:",
                    use_container_width=True,
                    key=f"menu-pin-{conversation_id}",
                ):

                    try:

                        api.update_conversation(
                            conversation_id,
                            pinned=not is_pinned,
                        )

                        st.rerun()

                    except APIError as exc:
                        fail(str(exc))


                # =================================================
                # SHARE
                # =================================================

                if st.button(
                    "Chia sẻ",
                    icon=":material/share:",
                    use_container_width=True,
                    key=f"share-{conversation_id}",
                ):

                    try:

                        token = api.share_conversation(
                            conversation_id
                        )

                        st.session_state.share_url = (
                            PUBLIC_BASE_URL.rstrip("/")
                            + "/?share="
                            + token
                        )

                        st.session_state.active_conversation_id = (
                            conversation_id
                        )

                        st.rerun()

                    except APIError as exc:
                        fail(str(exc))


                st.divider()


                # =================================================
                # DELETE
                # =================================================

                if st.button(
                    "Xóa",
                    icon=":material/delete:",
                    type="secondary",
                    use_container_width=True,
                    key=f"delete-{conversation_id}",
                ):

                    try:

                        api.delete_conversation(
                            conversation_id
                        )

                        if (
                            st.session_state.active_conversation_id
                            == conversation_id
                        ):
                            st.session_state.active_conversation_id = None


                        st.session_state.pop(
                            "share_url",
                            None,
                        )


                        st.rerun()

                    except APIError as exc:
                        fail(str(exc))


    # ============================================================
    # USER
    # ============================================================

    user = st.session_state.get("auth_user")


    if user:

        st.divider()

        st.caption(
            f"Đang đăng nhập: **{user.get('username', '')}**"
        )


        if st.button(
            "Đăng xuất",
            icon=":material/logout:",
            use_container_width=True,
            key="logout",
        ):

            try:
                api.logout()

            finally:

                st.session_state.clear()
                st.rerun()


    # ============================================================
    # API STATUS
    # ============================================================

    st.divider()

    online = api.health()

    st.caption(
        "● API online"
        if online
        else "● API unavailable"
    )


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
                    status_box.update(label="Câu trả lời bị gián đoạn, vui lòng thử lại", state="error")
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
