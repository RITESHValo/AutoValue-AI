
import streamlit as st
import pandas as pd
import numpy as np
import sqlite3
import hashlib
import hmac
import joblib
import plotly.express as px
from datetime import datetime
from pathlib import Path

# =====================================================
# AUTO VALUE AI - COMPLETE APPLICATION
# =====================================================

st.set_page_config(
    page_title="AutoValue AI",
    page_icon="🚘",
    layout="wide",
    initial_sidebar_state="expanded"
)

BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "users.db"
MODEL_PATH = BASE_DIR / "car_model.pkl"
DATA_PATH = BASE_DIR / "car_data_processed.csv"


# =====================================================
# PREMIUM UI
# =====================================================

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

.stApp {
    background: #080d19;
    font-family: 'Inter', sans-serif;
    color: #f8fafc;
}

[data-testid="stHeader"] {
    background: transparent;
}

[data-testid="stSidebar"] {
    background: #0d1527;
    border-right: 1px solid #202d45;
}

[data-testid="stSidebar"] * {
    color: #f1f5f9;
}

h1, h2, h3, h4, p, label,
[data-testid="stMarkdownContainer"],
[data-testid="stMetricLabel"],
[data-testid="stMetricValue"] {
    color: #f8fafc;
}

h1 {
    font-weight: 800 !important;
    letter-spacing: -1px;
}

h2, h3 {
    font-weight: 700 !important;
}

[data-testid="stForm"] {
    background: linear-gradient(145deg, #111c32, #0d1527);
    border: 1px solid #2c4164;
    border-radius: 20px;
    padding: 25px;
    box-shadow: 0 15px 45px rgba(0,0,0,.25);
    animation: loginAppear .4s ease-out;
}

.stFormSubmitButton button,
.stButton button {
    background: linear-gradient(135deg, #3b82f6, #6366f1) !important;
    color: white !important;
    border: 1px solid #668bff !important;
    border-radius: 12px !important;
    min-height: 44px;
    font-weight: 700 !important;
    transition: all .22s ease-in-out !important;
    box-shadow: 0 5px 18px rgba(59,130,246,.18);
}

.stFormSubmitButton button:hover,
.stButton button:hover {
    background: linear-gradient(135deg, #60a5fa, #818cf8) !important;
    transform: translateY(-2px);
    box-shadow: 0 9px 25px rgba(59,130,246,.35);
    border-color: #93c5fd !important;
}

.stFormSubmitButton button:active,
.stButton button:active {
    transform: scale(.98);
}

[data-testid="stTextInput"] input,
[data-testid="stNumberInput"] input {
    background: #0b1324 !important;
    color: white !important;
    border: 1px solid #30415e !important;
    border-radius: 11px !important;
    min-height: 42px;
}

[data-testid="stTextInput"] input:focus,
[data-testid="stNumberInput"] input:focus {
    border-color: #60a5fa !important;
    box-shadow: 0 0 0 3px rgba(59,130,246,.20) !important;
}

[data-baseweb="select"] > div {
    background: #0b1324 !important;
    border-color: #30415e !important;
    border-radius: 10px !important;
    color: white !important;
}

.stTabs [data-baseweb="tab-list"] {
    gap: 8px;
    border-bottom: 1px solid #263650;
}

.stTabs [data-baseweb="tab"] {
    background: #101b30;
    border-radius: 10px 10px 0 0;
    padding: 10px 20px;
    font-weight: 700;
}

.stTabs [aria-selected="true"] {
    background: #172744 !important;
    border-bottom: 3px solid #60a5fa !important;
}

[data-testid="stMetric"] {
    background: linear-gradient(145deg, #111c32, #0d1527);
    border: 1px solid #263650;
    border-radius: 16px;
    padding: 18px;
}

[data-testid="stMetricLabel"] {
    color: #9db0cb !important;
}

[data-testid="stMetricValue"] {
    color: #60a5fa !important;
    font-weight: 800;
}

hr {
    border-color: #263650;
}

[data-testid="stDataFrame"] {
    border: 1px solid #263650;
    border-radius: 12px;
}

div[data-testid="stAlert"] {
    border-radius: 12px;
}

@keyframes loginAppear {
    from {
        opacity: 0;
        transform: translateY(10px);
    }
    to {
        opacity: 1;
        transform: translateY(0);
    }
}
</style>
""", unsafe_allow_html=True)


# =====================================================
# DATABASE
# =====================================================

def get_connection():
    return sqlite3.connect(DB_PATH)


def init_database():
    with get_connection() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
        """)

        conn.execute("""
            CREATE TABLE IF NOT EXISTS prediction_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT NOT NULL,
                car_name TEXT NOT NULL,
                year INTEGER,
                km_driven INTEGER,
                predicted_price REAL,
                created_at TEXT NOT NULL
            )
        """)


def hash_password(password):
    return hashlib.sha256(password.encode("utf-8")).hexdigest()


def create_user(username, password):
    try:
        with get_connection() as conn:
            conn.execute("""
                INSERT INTO users (username, password_hash, created_at)
                VALUES (?, ?, ?)
            """, (
                username,
                hash_password(password),
                datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            ))
        return True
    except sqlite3.IntegrityError:
        return False


def verify_user(username, password):
    with get_connection() as conn:
        result = conn.execute("""
            SELECT password_hash
            FROM users
            WHERE username = ?
        """, (username,)).fetchone()

    if result is None:
        return False

    return hmac.compare_digest(
        result[0],
        hash_password(password)
    )


def save_prediction(username, car_name, year, km_driven, price):
    with get_connection() as conn:
        conn.execute("""
            INSERT INTO prediction_history
            (username, car_name, year, km_driven, predicted_price, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            username,
            car_name,
            int(year),
            int(km_driven),
            float(price),
            datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        ))


def get_user_history(username):
    with get_connection() as conn:
        return pd.read_sql_query("""
            SELECT car_name, year, km_driven,
                   predicted_price, created_at
            FROM prediction_history
            WHERE username = ?
            ORDER BY id DESC
        """, conn, params=(username,))


init_database()


# =====================================================
# LOAD MODEL AND DATA
# =====================================================

@st.cache_resource
def load_model_bundle():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "car_model.pkl nahi mila. Pehle train_model.py run kar."
        )

    bundle = joblib.load(MODEL_PATH)

    if not isinstance(bundle, dict) or "model" not in bundle:
        raise ValueError(
            "Model file ka format expected format se match nahi karta."
        )

    return bundle


@st.cache_data
def load_dataset():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            "car_data_processed.csv nahi mila."
        )

    return pd.read_csv(DATA_PATH)


try:
    model_bundle = load_model_bundle()
    car_data = load_dataset()

    model = model_bundle["model"]
    model_features = model_bundle.get(
        "features",
        [
            "brand", "model", "year", "km_driven",
            "fuel", "seller_type", "transmission", "owner"
        ]
    )

except Exception as error:
    st.error(f"Application load nahi ho paayi: {error}")
    st.info(
        "Check kar ki app.py, car_model.pkl aur "
        "car_data_processed.csv same folder mein hain."
    )
    st.stop()


# =====================================================
# HELPER FUNCTIONS
# =====================================================

def get_options(column, defaults):
    if column in car_data.columns:
        values = car_data[column].dropna().astype(str).unique().tolist()
        values = sorted(values)

        if values:
            return values

    return defaults


def format_inr(amount):
    return f"₹{amount:,.0f}"


def predict_price(input_values):
    input_df = pd.DataFrame([input_values])

    # Keep exactly the feature order used during training.
    input_df = input_df.reindex(columns=model_features)

    prediction = float(model.predict(input_df)[0])

    return max(0, prediction)


def section_heading(title, subtitle=None):
    st.markdown(f"## {title}")

    if subtitle:
        st.caption(subtitle)


# =====================================================
# LOGIN / SIGNUP
# =====================================================

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if "username" not in st.session_state:
    st.session_state.username = ""


if not st.session_state.logged_in:

    st.markdown("<div style='height:35px'></div>", unsafe_allow_html=True)

    left, center, right = st.columns([1, 1.15, 1])

    with center:
        st.markdown(
            """
            <div style="text-align:center; padding:12px 0 22px;">
                <div style="
                    display:inline-block;
                    padding:14px 18px;
                    border-radius:20px;
                    background:linear-gradient(135deg,#1d4ed8,#6366f1);
                    font-size:38px;
                    box-shadow:0 10px 35px rgba(59,130,246,.25);">
                    🚘
                </div>
                <h1 style="margin-top:18px; margin-bottom:4px;">
                    AutoValue <span style="color:#60a5fa;">AI</span>
                </h1>
                <p style="color:#9db0cb;">
                    Smart car valuation powered by Machine Learning
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )

        login_tab, signup_tab = st.tabs([
            "🔐 Login",
            "✨ Create Account"
        ])

        with login_tab:
            st.markdown("### Welcome back")
            st.caption("Login kar aur apna dashboard access kar.")

            with st.form("login_form"):
                login_username = st.text_input(
                    "Username",
                    placeholder="Enter your username"
                )

                login_password = st.text_input(
                    "Password",
                    type="password",
                    placeholder="Enter your password"
                )

                login_submit = st.form_submit_button(
                    "Login to Dashboard →",
                    use_container_width=True
                )

            if login_submit:
                username = login_username.strip()

                if not username or not login_password:
                    st.warning("Username aur password dono bhar.")
                elif verify_user(username, login_password):
                    st.session_state.logged_in = True
                    st.session_state.username = username
                    st.rerun()
                else:
                    st.error("Invalid username ya password.")

        with signup_tab:
            st.markdown("### Create your account")
            st.caption("Naya account bana aur AutoValue AI use kar.")

            with st.form("signup_form"):
                signup_username = st.text_input(
                    "Choose Username",
                    placeholder="Choose a username"
                )

                signup_password = st.text_input(
                    "Create Password",
                    type="password",
                    placeholder="At least 6 characters"
                )

                confirm_password = st.text_input(
                    "Confirm Password",
                    type="password",
                    placeholder="Enter password again"
                )

                signup_submit = st.form_submit_button(
                    "Create Account ✨",
                    use_container_width=True
                )

            if signup_submit:
                username = signup_username.strip()

                if len(username) < 3:
                    st.warning("Username minimum 3 characters ka rakh.")
                elif len(signup_password) < 6:
                    st.warning("Password minimum 6 characters ka rakh.")
                elif signup_password != confirm_password:
                    st.error("Dono passwords match nahi kar rahe.")
                else:
                    created = create_user(username, signup_password)

                    if created:
                        st.success(
                            "Account ban gaya! Ab Login tab se login kar."
                        )
                    else:
                        st.error(
                            "Ye username pehle se registered hai."
                        )

        st.markdown(
            """
            <p style="text-align:center; color:#7184a3;
            font-size:12px; margin-top:24px;">
                AutoValue AI · Intelligent Car Price Prediction
            </p>
            """,
            unsafe_allow_html=True
        )

    st.stop()


# =====================================================
# MAIN APP / SIDEBAR
# =====================================================

with st.sidebar:
    st.markdown("# 🚘 AutoValue AI")
    st.caption("Smart Car Price Prediction")

    st.divider()

    st.markdown("### 👋 Welcome")
    st.markdown(f"**{st.session_state.username}**")

    st.divider()

    page = st.radio(
        "Navigation",
        [
            "🏠 Overview",
            "💰 Price Predictor",
            "📊 Analytics Dashboard",
            "🕘 Prediction History"
        ],
        label_visibility="collapsed"
    )

    st.divider()

    st.caption("Machine Learning • Random Forest")

    if st.button("🚪 Logout", use_container_width=True):
        st.session_state.logged_in = False
        st.session_state.username = ""
        st.rerun()


# =====================================================
# OVERVIEW
# =====================================================

if page == "🏠 Overview":

    st.title("AutoValue AI")
    st.markdown(
        "### Your smart assistant for estimating used car prices."
    )

    st.markdown(
        """
        Estimate the market value of a used car using a trained
        Machine Learning model. Explore dataset trends and review
        your previous predictions from one dashboard.
        """
    )

    st.divider()

    total_cars = len(car_data)

    if "selling_price" in car_data.columns:
        average_price = car_data["selling_price"].mean()
        median_price = car_data["selling_price"].median()
    else:
        average_price = 0
        median_price = 0

    col1, col2, col3 = st.columns(3)

    col1.metric("🚗 Dataset Records", f"{total_cars:,}")
    col2.metric("💰 Average Car Price", format_inr(average_price))
    col3.metric("📍 Median Car Price", format_inr(median_price))

    st.markdown("## What can you do?")

    c1, c2 = st.columns(2)

    with c1:
        st.markdown("### 💰 Price Predictor")
        st.write(
            "Car ki details enter kar aur estimated selling price "
            "check kar."
        )

    with c2:
        st.markdown("### 📊 Analytics Dashboard")
        st.write(
            "Fuel type, transmission, car prices aur dataset ke "
            "patterns explore kar."
        )

    st.info(
        "Prediction ek estimate hai. Actual selling price car ki "
        "condition, location aur market demand par depend kar sakti hai."
    )


# =====================================================
# PRICE PREDICTOR
# =====================================================

elif page == "💰 Price Predictor":

    st.title("💰 Car Price Predictor")
    st.caption(
        "Enter the car details below to estimate its selling price."
    )

    brand_options = get_options(
        "brand",
        ["Maruti", "Hyundai", "Honda", "Tata", "Mahindra"]
    )

    fuel_options = get_options(
        "fuel",
        ["Petrol", "Diesel", "CNG", "LPG", "Electric"]
    )

    seller_options = get_options(
        "seller_type",
        ["Individual", "Dealer", "Trustmark Dealer"]
    )

    transmission_options = get_options(
        "transmission",
        ["Manual", "Automatic"]
    )

    owner_options = get_options(
        "owner",
        [
            "First Owner",
            "Second Owner",
            "Third Owner",
            "Fourth & Above Owner",
            "Test Drive Car"
        ]
    )

    with st.form("predictor_form"):

        st.markdown("### 🚘 Vehicle Details")

        col1, col2 = st.columns(2)

        with col1:
            brand = st.selectbox(
                "Car Brand",
                brand_options
            )

            year = st.number_input(
                "Manufacturing Year",
                min_value=1990,
                max_value=datetime.now().year,
                value=2020,
                step=1
            )

            km_driven = st.number_input(
                "Kilometres Driven",
                min_value=0,
                max_value=1_000_000,
                value=30000,
                step=1000
            )

            fuel = st.selectbox(
                "Fuel Type",
                fuel_options
            )

        with col2:
            model_options = []

            if "brand" in car_data.columns and "model" in car_data.columns:
                model_options = (
                    car_data.loc[
                        car_data["brand"].astype(str) == str(brand),
                        "model"
                    ]
                    .dropna()
                    .astype(str)
                    .unique()
                    .tolist()
                )

            model_options = sorted(model_options)

            if not model_options:
                model_options = get_options(
                    "model",
                    ["Other"]
                )

            car_model_name = st.selectbox(
                "Car Model",
                model_options
            )

            seller_type = st.selectbox(
                "Seller Type",
                seller_options
            )

            transmission = st.selectbox(
                "Transmission",
                transmission_options
            )

            owner = st.selectbox(
                "Owner Type",
                owner_options
            )

        submitted = st.form_submit_button(
            "✨ Predict Car Price",
            use_container_width=True
        )

    if submitted:

        input_values = {
            "brand": brand,
            "model": car_model_name,
            "year": int(year),
            "km_driven": int(km_driven),
            "fuel": fuel,
            "seller_type": seller_type,
            "transmission": transmission,
            "owner": owner
        }

        try:
            estimated_price = predict_price(input_values)

            st.divider()
            st.markdown("## 🎯 Estimated Selling Price")

            st.markdown(
                f"""
                <div style="
                    background:linear-gradient(135deg,#102b4e,#172554);
                    border:1px solid #3b82f6;
                    padding:28px;
                    border-radius:20px;
                    text-align:center;
                    margin:12px 0 20px;">
                    <div style="color:#bfdbfe; font-size:15px;">
                        Predicted Market Value
                    </div>
                    <div style="
                        font-size:42px;
                        font-weight:800;
                        color:#60a5fa;
                        margin-top:8px;">
                        {format_inr(estimated_price)}
                    </div>
                    <div style="color:#a5b4fc; font-size:13px;
                    margin-top:8px;">
                        AI-powered estimate
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

            save_prediction(
                st.session_state.username,
                f"{brand} {car_model_name}",
                year,
                km_driven,
                estimated_price
            )

            st.success("Prediction history mein save ho gayi.")

            st.caption(
                "Note: Ye machine-learning estimate hai, guaranteed "
                "resale price nahi."
            )

        except Exception as error:
            st.error(f"Prediction nahi ho paayi: {error}")
            st.info(
                "Check kar ki model training ke features aur "
                "car_data_processed.csv ke columns match karte hain."
            )


# =====================================================
# ANALYTICS DASHBOARD
# =====================================================

elif page == "📊 Analytics Dashboard":

    st.title("📊 Analytics Dashboard")
    st.caption("Explore trends in the used-car dataset.")

    df = car_data.copy()

    if "selling_price" not in df.columns:
        st.error("Dataset mein selling_price column nahi mila.")
        st.stop()

    filter_col1, filter_col2 = st.columns(2)

    if "fuel" in df.columns:
        fuel_values = sorted(
            df["fuel"].dropna().astype(str).unique().tolist()
        )

        with filter_col1:
            selected_fuels = st.multiselect(
                "Filter by Fuel",
                fuel_values,
                default=fuel_values
            )

        df = df[df["fuel"].astype(str).isin(selected_fuels)]

    if "transmission" in df.columns:
        transmission_values = sorted(
            df["transmission"].dropna().astype(str).unique().tolist()
        )

        with filter_col2:
            selected_transmissions = st.multiselect(
                "Filter by Transmission",
                transmission_values,
                default=transmission_values
            )

        df = df[
            df["transmission"].astype(str).isin(selected_transmissions)
        ]

    st.divider()

    if df.empty:
        st.warning("Selected filters ke liye koi records nahi mile.")
        st.stop()

    avg_price = df["selling_price"].mean()
    median_price = df["selling_price"].median()
    total_records = len(df)

    m1, m2, m3 = st.columns(3)

    m1.metric("Filtered Records", f"{total_records:,}")
    m2.metric("Average Price", format_inr(avg_price))
    m3.metric("Median Price", format_inr(median_price))

    st.divider()

    chart_left, chart_right = st.columns(2)

    with chart_left:
        st.markdown("### 💰 Price Distribution")

        fig_price = px.histogram(
            df,
            x="selling_price",
            nbins=30,
            title="Distribution of Selling Prices",
            labels={"selling_price": "Selling Price (₹)"},
            template="plotly_dark"
        )

        fig_price.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)"
        )

        st.plotly_chart(fig_price, use_container_width=True)

    with chart_right:
        if "fuel" in df.columns:
            st.markdown("### ⛽ Average Price by Fuel")

            fuel_summary = (
                df.groupby("fuel", as_index=False)["selling_price"]
                .mean()
                .sort_values("selling_price", ascending=False)
            )

            fig_fuel = px.bar(
                fuel_summary,
                x="fuel",
                y="selling_price",
                title="Average Selling Price by Fuel Type",
                labels={
                    "fuel": "Fuel Type",
                    "selling_price": "Average Price (₹)"
                },
                template="plotly_dark"
            )

            fig_fuel.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)"
            )

            st.plotly_chart(fig_fuel, use_container_width=True)

    chart_left, chart_right = st.columns(2)

    with chart_left:
        if "year" in df.columns:
            st.markdown("### 📅 Price by Manufacturing Year")

            year_summary = (
                df.groupby("year", as_index=False)["selling_price"]
                .mean()
                .sort_values("year")
            )

            fig_year = px.line(
                year_summary,
                x="year",
                y="selling_price",
                markers=True,
                title="Average Price by Year",
                labels={
                    "year": "Manufacturing Year",
                    "selling_price": "Average Price (₹)"
                },
                template="plotly_dark"
            )

            fig_year.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)"
            )

            st.plotly_chart(fig_year, use_container_width=True)

    with chart_right:
        if "transmission" in df.columns:
            st.markdown("### ⚙️ Transmission Breakdown")

            transmission_summary = (
                df["transmission"]
                .astype(str)
                .value_counts()
                .rename_axis("transmission")
                .reset_index(name="count")
            )

            fig_transmission = px.pie(
                transmission_summary,
                names="transmission",
                values="count",
                hole=0.48,
                title="Cars by Transmission",
                template="plotly_dark"
            )

            fig_transmission.update_layout(
                paper_bgcolor="rgba(0,0,0,0)"
            )

            st.plotly_chart(
                fig_transmission,
                use_container_width=True
            )

    st.divider()

    st.markdown("### 📋 Filtered Dataset Preview")

    st.dataframe(
        df.head(100),
        use_container_width=True,
        hide_index=True
    )

    csv_data = df.to_csv(index=False).encode("utf-8")

    st.download_button(
        "⬇️ Download Filtered Dataset (CSV)",
        data=csv_data,
        file_name="autovalue_filtered_dataset.csv",
        mime="text/csv",
        use_container_width=True
    )


# =====================================================
# PREDICTION HISTORY
# =====================================================

elif page == "🕘 Prediction History":

    st.title("🕘 Prediction History")
    st.caption("Teri previous car price predictions.")

    history = get_user_history(st.session_state.username)

    if history.empty:
        st.info(
            "Abhi koi prediction history nahi hai. "
            "Price Predictor se pehli prediction kar."
        )

    else:
        history.columns = [
            "Car",
            "Manufacturing Year",
            "Kilometres Driven",
            "Predicted Price (₹)",
            "Date & Time"
        ]

        h1, h2, h3 = st.columns(3)

        h1.metric("Total Predictions", len(history))

        h2.metric(
            "Average Predicted Price",
            format_inr(history["Predicted Price (₹)"].mean())
        )

        h3.metric(
            "Latest Prediction",
            format_inr(history.iloc[0]["Predicted Price (₹)"])
        )

        st.divider()

        display_history = history.copy()

        display_history["Predicted Price (₹)"] = (
            display_history["Predicted Price (₹)"].map(format_inr)
        )

        st.dataframe(
            display_history,
            use_container_width=True,
            hide_index=True
        )

        original_history = history.copy()

        st.download_button(
            "⬇️ Download My Prediction History",
            data=original_history.to_csv(index=False).encode("utf-8"),
            file_name="my_prediction_history.csv",
            mime="text/csv",
            use_container_width=True
        )


# =====================================================
# FOOTER
# =====================================================

st.divider()

st.markdown(
    """
    <div style="text-align:center; padding:10px 0 20px;">
        <p style="color:#7184a3; font-size:12px;">
            AutoValue AI · Machine Learning Based Used Car Price Prediction
        </p>
    </div>
    """,
    unsafe_allow_html=True
)
