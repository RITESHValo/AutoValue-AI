
import streamlit as st
import sqlite3
import hashlib
import secrets
from pathlib import Path
import re

DB_PATH = Path(__file__).with_name("users.db")
ITERATIONS = 300_000


# ---------- DATABASE ----------
def init_db():
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                salt TEXT NOT NULL,
                password_hash TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)


# ---------- PASSWORD SECURITY ----------
def hash_password(password, salt):
    return hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        bytes.fromhex(salt),
        ITERATIONS
    ).hex()


def create_user(username, password):
    username = username.strip()

    if len(username) < 3:
        return False, "Username must contain at least 3 characters."

    if not re.fullmatch(r"[A-Za-z0-9_.-]+", username):
        return False, "Use only letters, numbers, dots, underscores or hyphens."

    if len(password) < 8:
        return False, "Password must contain at least 8 characters."

    salt = secrets.token_hex(16)
    password_hash = hash_password(password, salt)

    try:
        with sqlite3.connect(DB_PATH) as conn:
            conn.execute(
                "INSERT INTO users (username, salt, password_hash) VALUES (?, ?, ?)",
                (username, salt, password_hash)
            )
        return True, "Account created successfully! Please log in."

    except sqlite3.IntegrityError:
        return False, "This username already exists. Please choose another."


def verify_user(username, password):
    with sqlite3.connect(DB_PATH) as conn:
        user = conn.execute(
            "SELECT salt, password_hash FROM users WHERE username = ?",
            (username.strip(),)
        ).fetchone()

    if user is None:
        return False

    salt, saved_hash = user
    entered_hash = hash_password(password, salt)

    return secrets.compare_digest(entered_hash, saved_hash)


# ---------- AUTH PAGE ----------
def show_auth():
    init_db()

    # Force all login/signup text to white
    st.markdown("""
    <style>
    /* Main page */
    .stApp,
    [data-testid="stAppViewContainer"],
    [data-testid="stMain"],
    [data-testid="stMainBlockContainer"] {
        background-color: #0b1120 !important;
        color: #ffffff !important;
    }

    /* Every text element */
    .stApp p,
    .stApp label,
    .stApp span,
    .stApp div,
    .stApp h1,
    .stApp h2,
    .stApp h3,
    .stApp h4,
    .stApp h5,
    .stApp h6,
    .stApp li,
    .stApp small,
    .stApp strong,
    .stApp [data-testid="stMarkdownContainer"],
    .stApp [data-testid="stWidgetLabel"],
    .stApp [data-testid="stCaptionContainer"] {
        color: #ffffff !important;
    }

    /* Tabs */
    .stApp button[data-baseweb="tab"],
    .stApp button[data-baseweb="tab"] *,
    .stApp [role="tab"],
    .stApp [role="tab"] * {
        color: #ffffff !important;
    }

    /* Input boxes */
    .stApp input,
    .stApp textarea,
    .stApp [data-baseweb="input"] input {
        color: #ffffff !important;
        background-color: #151e30 !important;
        -webkit-text-fill-color: #ffffff !important;
        caret-color: #ffffff !important;
    }

    /* Input placeholders */
    .stApp input::placeholder,
    .stApp textarea::placeholder {
        color: #ffffff !important;
        opacity: 1 !important;
        -webkit-text-fill-color: #ffffff !important;
    }

    /* Password fields and input wrappers */
    .stApp [data-baseweb="input"],
    .stApp [data-baseweb="base-input"] {
        background-color: #151e30 !important;
    }

    /* Buttons */
    .stApp button[kind="primary"] {
        background-color: #2563eb !important;
        border: 1px solid #3b82f6 !important;
    }

    .stApp button[kind="primary"],
    .stApp button[kind="primary"] * {
        color: #ffffff !important;
        -webkit-text-fill-color: #ffffff !important;
    }

    /* Alerts and messages */
    .stApp [data-testid="stAlert"],
    .stApp [data-testid="stAlert"] * {
        color: #ffffff !important;
    }

    /* Remove top header strip */
    header[data-testid="stHeader"] {
        background-color: transparent !important;
    }

    /* Hide menu and footer */
    #MainMenu,
    footer {
        visibility: hidden;
    }
    </style>
    """, unsafe_allow_html=True)

    if st.session_state.get("authenticated", False):
        return True

    # Login card
    st.markdown("""
    <div style="
        max-width:520px;
        margin:45px auto 20px auto;
        padding:25px 10px 10px 10px;
        text-align:center;
    ">
        <h1 style="
            color:#ffffff !important;
            font-size:38px;
            font-weight:800;
            margin-bottom:5px;
        ">🚘 AutoValue AI</h1>
        <p style="
            color:#ffffff !important;
            font-size:16px;
            margin-top:0;
        ">Smart Car Price Prediction Platform</p>
    </div>
    """, unsafe_allow_html=True)

    _, center, _ = st.columns([1, 2, 1])

    with center:
        login_tab, signup_tab = st.tabs(["Login", "Sign Up"])

        with login_tab:
            st.markdown("### Welcome Back")
            st.markdown("Log in to continue to AutoValue AI.")

            with st.form("login_form"):
                username = st.text_input(
                    "Username",
                    placeholder="Enter your username"
                )
                password = st.text_input(
                    "Password",
                    type="password",
                    placeholder="Enter your password"
                )
                login_button = st.form_submit_button(
                    "Login",
                    use_container_width=True
                )

            if login_button:
                if verify_user(username, password):
                    st.session_state.authenticated = True
                    st.session_state.username = username.strip()
                    st.rerun()
                else:
                    st.error(
                        "Invalid username or password. Please try again."
                    )

        with signup_tab:
            st.markdown("### Create Your Account")
            st.markdown("Sign up to get started.")

            with st.form("signup_form"):
                new_username = st.text_input(
                    "Choose Username",
                    placeholder="Create a username"
                )
                new_password = st.text_input(
                    "Create Password",
                    type="password",
                    placeholder="Minimum 8 characters"
                )
                confirm_password = st.text_input(
                    "Confirm Password",
                    type="password",
                    placeholder="Re-enter your password"
                )

                signup_button = st.form_submit_button(
                    "Create Account",
                    use_container_width=True
                )

            if signup_button:
                if new_password != confirm_password:
                    st.error("Passwords do not match.")
                else:
                    success, message = create_user(
                        new_username, new_password
                    )

                    if success:
                        st.success(message)
                    else:
                        st.error(message)

    return False
