from __future__ import annotations

import streamlit as st

from api_client import AdminAPI, APIError


st.set_page_config(
    page_title="LawChat Admin",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# API CLIENT
# ============================================================

if "admin_api" not in st.session_state:
    st.session_state.admin_api = AdminAPI()

api: AdminAPI = st.session_state.admin_api


# ============================================================
# SESSION STATE
# ============================================================

if "auth_user" not in st.session_state:
    st.session_state.auth_user = None

if "login_username" not in st.session_state:
    st.session_state.login_username = ""

if "login_password" not in st.session_state:
    st.session_state.login_password = ""


# ============================================================
# STYLING
# ============================================================

st.markdown(
    """
    <style>
        .stApp {
            background-color: #0b0f14;
        }

        .admin-brand {
            font-size: 24px;
            font-weight: 700;
            letter-spacing: 1px;
        }

        .admin-subtitle {
            color: #8b949e;
            margin-bottom: 24px;
        }

        .admin-card {
            background: #111820;
            border: 1px solid #263241;
            border-radius: 12px;
            padding: 20px;
            min-height: 120px;
        }

        .admin-card-title {
            color: #8b949e;
            font-size: 14px;
            margin-bottom: 8px;
        }

        .admin-card-value {
            font-size: 28px;
            font-weight: 700;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# LOGIN
# ============================================================

def login_screen() -> None:
    st.markdown(
        '<div class="admin-brand">⚙️ LAWCHAT ADMIN</div>',
        unsafe_allow_html=True,
    )

    st.caption("Administrator authentication")

    st.divider()

    with st.form("admin_login_form"):
        password = st.text_input(
            "Password",
            type="password",
        )

        submitted = st.form_submit_button(
            "Continue",
            use_container_width=True,
        )

    if not submitted:
        return

    try:
        username = "admin"

        result = api.login(
            username=username,
            password=password,
        )

        st.session_state.login_username = username
        st.session_state.login_password = password

        if result.get("requires_totp"):
            st.session_state.auth_stage = "totp"
            st.rerun()

        st.error("Unexpected authentication response.")

    except APIError as exc:
        st.error(str(exc))


# ============================================================
# TOTP
# ============================================================

def totp_screen() -> None:
    st.markdown(
        '<div class="admin-brand">🔐 Two-Factor Authentication</div>',
        unsafe_allow_html=True,
    )

    st.caption(
        "Enter the verification code from your authenticator app."
    )

    st.divider()

    with st.form("admin_totp_form"):
        code = st.text_input(
            "Authentication code",
            max_chars=6,
            placeholder="000000",
        )

        submitted = st.form_submit_button(
            "Verify and sign in",
            use_container_width=True,
        )

    if not submitted:
        return

    try:
        user = api.verify_totp(
            username=st.session_state.login_username,
            password=st.session_state.login_password,
            code=code,
        )

        if user.get("role") != "ADMIN":
            st.error("You are not authorized to access the admin panel.")
            api.logout()
            return

        st.session_state.auth_user = user
        st.session_state.login_password = ""
        st.session_state.auth_stage = "authenticated"

        st.rerun()

    except APIError as exc:
        st.error(str(exc))


# ============================================================
# AUTH CHECK
# ============================================================

def require_admin() -> bool:
    user = st.session_state.get("auth_user")

    if user is not None:
        if user.get("role") != "ADMIN":
            st.error("Access denied.")
            st.stop()

        return True

    try:
        user = api.me()

        if user.get("role") != "ADMIN":
            st.error("Access denied.")
            st.stop()

        st.session_state.auth_user = user
        return True

    except APIError:
        pass

    return False


# ============================================================
# ADMIN DASHBOARD
# ============================================================

def dashboard() -> None:
    user = st.session_state.auth_user

    with st.sidebar:
        st.markdown(
            '<div class="admin-brand">⚙️ LawChat Admin</div>',
            unsafe_allow_html=True,
        )

        st.caption("System Administration")

        st.divider()

        page = st.radio(
            "Navigation",
            [
                "📊 Dashboard",
                "👥 Users",
                "🔐 Security",
                "📚 RAG / Data",
                "⚙️ System",
            ],
            label_visibility="collapsed",
        )

        st.divider()

        st.caption(f"Signed in as: {user.get('username', 'admin')}")
        
       
        st.divider()

        if st.button(
            "Test Admin API",
            use_container_width=True,
        ):
            try:
                result = api._request(
                    "GET",
                    "/api/v1/admin/ping",
                )
                st.success(
                    f"Backend OK: {result['username']} / {result['role']}"
                )
            except APIError as exc:
                st.error(str(exc))
        if st.button(
            "Logout",
            use_container_width=True,
        ):
            try:
                api.logout()
            except APIError:
                pass

            st.session_state.auth_user = None
            st.session_state.auth_stage = "login"
            st.rerun()

    st.markdown(
        '<div class="admin-brand">LAWCHAT ADMIN PANEL</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="admin-subtitle">'
        "Administrative control center for LawChat"
        "</div>",
        unsafe_allow_html=True,
    )

    if page == "📊 Dashboard":
        st.subheader("System Overview")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown(
                """
                <div class="admin-card">
                    <div class="admin-card-title">
                        API / Backend
                    </div>
                    <div class="admin-card-value">
                        🟢 Online
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with col2:
            st.markdown(
                """
                <div class="admin-card">
                    <div class="admin-card-title">
                        PostgreSQL
                    </div>
                    <div class="admin-card-value">
                        🟢 Online
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with col3:
            st.markdown(
                """
                <div class="admin-card">
                    <div class="admin-card-title">
                        RAG / Qdrant
                    </div>
                    <div class="admin-card-value">
                        🟢 Online
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    elif page == "👥 Users":
        st.subheader("User Management")
        st.caption("Manage user accounts and monthly token quotas.")

        try:
            users = api.list_users()
        except APIError as exc:
            st.error(f"Failed to load users: {exc}")
            return

        normal_users = [
            item for item in users
            if item.get("role") == "USER"
        ]

        active_users = [
            item for item in normal_users
            if item.get("is_active")
        ]

        total_used = sum(
            int(item.get("tokens_used") or 0)
            for item in normal_users
        )

        total_limit = sum(
            int(item.get("token_limit") or 0)
            for item in normal_users
        )

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Users", len(normal_users))

        with col2:
            st.metric("Active Users", len(active_users))

        with col3:
            st.metric("Tokens Used", f"{total_used:,}")

        st.divider()

        st.markdown("### User Quotas")

        if not normal_users:
            st.info("No USER accounts found.")
        else:
            for item in normal_users:
                username = item.get("username", "")
                user_id = item.get("id")

                token_limit = int(item.get("token_limit") or 0)
                tokens_used = int(item.get("tokens_used") or 0)
                remaining = int(item.get("remaining_tokens") or 0)

                period_start = item.get("period_start")
                period_end = item.get("period_end")

                with st.container(border=True):
                    col_user, col_used, col_limit, col_remaining = st.columns(
                        [2.2, 1.5, 1.5, 1.5]
                    )

                with col_user:
                    st.markdown(f"**{username}**")
                    st.caption(
                        "Active" if item.get("is_active") else "Inactive"
                    )

                with col_used:
                    st.metric("Used", f"{tokens_used:,}")

                with col_limit:
                    st.metric("Limit", f"{token_limit:,}")

                with col_remaining:
                    st.metric("Remaining", f"{remaining:,}")

                if period_start and period_end:
                    st.caption(
                        f"Quota period: {period_start} → {period_end}"
                    )

                with st.expander("Edit token quota"):
                    with st.form(f"quota_form_{user_id}"):
                        new_limit = st.number_input(
                            "Monthly token limit",
                            min_value=0,
                            max_value=10_000_000_000,
                            value=token_limit,
                            step=10_000,
                            key=f"quota_input_{user_id}",
                        )

                        save_quota = st.form_submit_button(
                            "Save quota",
                            use_container_width=True,
                        )

                    if save_quota:
                        try:
                            updated = api.update_user_usage(
                                user_id=str(user_id),
                                token_limit=int(new_limit),
                            )

                            st.success(
                                f"Quota updated for {username}: "
                                f"{int(updated.get('token_limit', new_limit)):,} tokens."
                            )

                            st.rerun()

                        except APIError as exc:
                            st.error(f"Failed to update quota: {exc}")

        if normal_users:
            st.divider()

            st.markdown("### Quota Summary")

            summary_col1, summary_col2 = st.columns(2)

            with summary_col1:
                st.metric("Total Allocated", f"{total_limit:,}")

            with summary_col2:
                st.metric("Total Used", f"{total_used:,}")

    elif page == "🔐 Security":
        st.subheader("Security & Audit")
        st.info("Security and audit controls will be connected next.")

    elif page == "📚 RAG / Data":
        st.subheader("RAG / Data")
        st.info("RAG monitoring and data controls will be connected next.")

    elif page == "⚙️ System":
        st.subheader("System Configuration")
        st.info("System configuration will be connected next.")


# ============================================================
# MAIN
# ============================================================

auth_stage = st.session_state.get("auth_stage", "login")

if auth_stage == "authenticated":
    if require_admin():
        dashboard()
    else:
        st.session_state.auth_stage = "login"
        login_screen()

elif auth_stage == "totp":
    totp_screen()

else:
    login_screen()
