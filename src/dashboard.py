"""Dashboard: a chat box that opens with a short intro to the data.

Questions go through the same pipeline as the command-line app (app.answer).
Run from the project folder with:

    streamlit run src/dashboard.py
"""

import logging

import altair as alt
import pandas as pd
import streamlit as st

from config import Config
from db import connect_readonly

st.set_page_config(page_title="Olist data chat", layout="wide")

ROBOT = Config.BASE_DIR / "assets" / "robot.svg"  # the assistant's avatar
USER = Config.BASE_DIR / "assets" / "user.svg"  # the user's avatar

log = logging.getLogger("dashboard")


def table_columns(conn) -> dict[str, list[str]]:
    """Every table and its column names, read from the database itself."""
    names = [
        r[0]
        for r in conn.execute(
            "SELECT name FROM sqlite_master "
            "WHERE type = 'table' AND name NOT LIKE 'sqlite_%'"
        )
    ]
    order = ["orders", "customers", "order_reviews"]  # the main table first
    names.sort(key=lambda n: order.index(n) if n in order else len(order))
    return {n: [r[1] for r in conn.execute(f'PRAGMA table_info("{n}")')] for n in names}


@st.cache_data
def load_stats() -> dict:
    """Read a few summary numbers from the database (read-only), once."""
    conn = connect_readonly(Config.DB_PATH)
    try:
        one = lambda sql: conn.execute(sql).fetchone()[0]
        first, last = conn.execute(
            "SELECT MIN(order_purchase_timestamp), MAX(order_purchase_timestamp) FROM orders"
        ).fetchone()
        return {
            "orders": one("SELECT COUNT(*) FROM orders"),
            "customers": one(
                "SELECT COUNT(DISTINCT customer_unique_id) FROM customers"
            ),
            "reviews": one("SELECT COUNT(*) FROM order_reviews"),
            "first": first[:10],
            "last": last[:10],
            "columns": table_columns(conn),
            "status": pd.read_sql_query(
                "SELECT order_status, COUNT(*) AS orders FROM orders "
                "GROUP BY order_status ORDER BY orders DESC",
                conn,
            ).set_index("order_status"),
            "monthly": pd.read_sql_query(
                "SELECT strftime('%Y-%m', order_purchase_timestamp) AS month, "
                "COUNT(*) AS orders FROM orders GROUP BY month ORDER BY month",
                conn,
            ).set_index("month"),
        }
    finally:
        conn.close()


@st.cache_resource
def load_pipeline():
    """Import the pipeline and build its system prompt, once.

    Imported here and not at the top of the file: llm.py raises if the API key
    is missing, and that should show up as a chat message, not stop the page.
    """
    import app

    conn = connect_readonly(Config.DB_PATH)
    try:
        system = app.build_system_prompt(app.describe_schema(conn))
    finally:
        conn.close()
    return app, system


def get_answer(question: str) -> tuple[str, str, str, list[str]]:
    """Run one question: (answer text, SQL used, assumption, suggested questions)."""
    try:
        app, system = load_pipeline()
        # A new read-only connection per question: SQLite connections can't be
        # shared between the threads Streamlit runs the page in.
        conn = connect_readonly(Config.DB_PATH)
        try:
            result = app.answer(conn, system, question)
        finally:
            conn.close()
    except RuntimeError as e:  # llm.py: the API key is not set
        return f"The model isn't available: {e}", "", "", []
    except Exception:
        log.exception("Failed to answer: %s", question)
        return "Something went wrong answering that. Please try again.", "", "", []
    return result.message, result.sql, result.assumption, result.suggestions


def monthly_chart(monthly: pd.DataFrame) -> alt.Chart:
    """Orders per month. Not .interactive(): the mouse wheel must scroll the page."""
    data = monthly.reset_index()
    data["month"] = pd.to_datetime(data["month"])
    return (
        alt.Chart(data)
        .mark_line(point=True)
        .encode(
            x=alt.X("month:T", title=None, axis=alt.Axis(format="%b %Y")),
            y=alt.Y("orders:Q", title=None),
            tooltip=[alt.Tooltip("month:T", format="%B %Y"), "orders:Q"],
        )
        .properties(height=220)
    )


def status_chart(status: pd.DataFrame) -> alt.Chart:
    """Horizontal bars, largest first. Fixed, so the page scrolls over it."""
    return (
        alt.Chart(status.reset_index())
        .mark_bar()
        .encode(
            x=alt.X("orders:Q", title=None),
            y=alt.Y("order_status:N", sort="-x", title=None),
            tooltip=["order_status:N", "orders:Q"],
        )
        .properties(height=220)
    )


# What one row means in each table: the part a column name alone does not tell you.
TABLE_NOTES = {
    "orders": " (one row per order)",
    "customers": " (one row per order's buyer; one person can appear in several rows)",
    "order_reviews": " (one row per review; an order can have more than one)",
}


def welcome_message(stats: dict) -> str:
    """The first chat message: what the data is and how to ask about it."""
    table_lines = "\n".join(
        f"- **{name}**{TABLE_NOTES.get(name, '')}: " + ", ".join(f"`{c}`" for c in cols)
        for name, cols in stats["columns"].items()
    )
    return f"""**Hi, I am Capi! Ask me anything about this data.**

It is e-commerce order data from Olist (Brazil): **{stats["orders"]:,} orders** \
from **{stats["customers"]:,} customers**, placed between {stats["first"]} and \
{stats["last"]}, with {stats["reviews"]:,} reviews.

**What's in the data** (every table and its columns)
{table_lines}

**Not in the data:** prices, revenue, payments, products, sellers, customer age or gender. \
I can't answer questions about those.

**How to ask questions**
- Use plain English and ask one question at a time.
- Be specific. Words like "best" or "late" can mean several things, so say which: \
"customers with the most orders", or "delivered after the estimated delivery date".
- The data ends in {stats["last"][:7]}, so "last month" means the latest month in the data.
- Per-person questions count each customer once (`customer_unique_id`), because \
`customer_id` changes with every order.
"""


def avatar(role: str) -> str | None:
    """Robot for the assistant, green person for the user.

    Falls back to Streamlit's default avatar if an image file is missing.
    """
    image = ROBOT if role == "assistant" else USER
    return str(image) if image.exists() else None


def ask_this(question: str) -> None:
    """Button callback: queue the question; show_chat picks it up on the rerun."""
    st.session_state.pending = question


def show_body(
    content: str,
    sql: str = "",
    assumption: str = "",
    suggestions: tuple | list = (),
    key: str = "",
) -> None:
    """A message's text, how the question was read, the SQL (closed at first)
    and suggested questions as buttons. key keeps the buttons' ids unique."""
    st.markdown(content)
    if assumption:
        st.caption(f"How I read your question: {assumption}")
    if sql:
        with st.expander("SQL used"):
            st.code(sql, language="sql")
    if suggestions:
        st.caption("You could ask:")
        for i, suggestion in enumerate(suggestions):
            st.button(
                suggestion,
                key=f"{key}-{i}",
                on_click=ask_this,
                args=(suggestion,),
                width="stretch",
            )


def pin_overview_css() -> str:
    """CSS that keeps the overview bar in view while the chat scrolls underneath.

    The bar needs a solid background or the chat would show through it.
    Opened, it scrolls inside itself so it never covers more than 60% of the screen.
    """
    # The wrapper Streamlit puts around the keyed container is what has to stick:
    # it sits directly inside the tall page column, so it has room to move.
    # light-dark() follows Streamlit's live color-scheme, so the background changes
    # the moment the theme is switched (Python only runs on a page reload).
    return """<style>
[data-testid="stLayoutWrapper"]:has(> .st-key-overview) {
    position: sticky; top: 3.75rem; z-index: 99;
    background: light-dark(#ffffff, #0e1117); padding-bottom: 0.25rem;
}
.st-key-overview [data-testid="stExpanderDetails"] {
    max-height: 60vh; overflow-y: auto;
}
</style>"""


def show_overview(stats: dict) -> None:
    """Key numbers and two charts, collapsed and pinned so it is always at hand."""
    st.markdown(pin_overview_css(), unsafe_allow_html=True)
    status = stats["status"]["orders"]
    delivered = status.get("delivered", 0)
    total = status.sum()

    with st.container(key="overview"), st.expander("Data overview"):
        st.caption(f"Olist e-commerce orders, {stats['first']} to {stats['last']}")
        cols = st.columns(4)
        cols[0].metric("Orders", f"{stats['orders']:,}")
        cols[1].metric("Unique customers", f"{stats['customers']:,}")
        cols[2].metric("Reviews", f"{stats['reviews']:,}")
        cols[3].metric("Delivered", f"{delivered / total:.1%}")

        left, right = st.columns([2, 1])
        left.caption("Orders per month")
        left.altair_chart(monthly_chart(stats["monthly"]), width="stretch")
        # Delivered is ~97% of orders and would flatten every other bar, so it
        # is shown as the metric above and left out of the chart.
        right.caption("Orders not delivered, by status")
        right.altair_chart(
            status_chart(stats["status"].drop(index="delivered", errors="ignore")),
            width="stretch",
        )


def show_chat(stats: dict) -> None:
    """Chat history plus the input box; each new question gets one reply."""
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": welcome_message(stats),
                "suggestions": list(Config.EXAMPLE_QUESTIONS),
            }
        ]
    messages = st.session_state.messages

    # A clicked suggestion is queued in session state and handled like typed input.
    typed = st.chat_input("Ask a question about the data...")
    question = st.session_state.pop("pending", None) or typed

    for i, message in enumerate(messages):
        # Suggestion buttons only under the newest message, and not while a new
        # question is being answered (they would be stale).
        show_buttons = i == len(messages) - 1 and not question
        with st.chat_message(message["role"], avatar=avatar(message["role"])):
            show_body(
                message["content"],
                message.get("sql", ""),
                message.get("assumption", ""),
                message.get("suggestions", ()) if show_buttons else (),
                key=f"s{i}",
            )

    if not question:
        return
    # Show the question straight away, then the reply once it is ready.
    with st.chat_message("user", avatar=avatar("user")):
        st.markdown(question)
    with st.chat_message("assistant", avatar=avatar("assistant")):
        with st.spinner("Thinking..."):
            reply, sql, assumption, suggestions = get_answer(question)
        # The reply will sit at index len(messages) + 1, so its button ids match
        # the ones the history loop uses on the next run.
        show_body(reply, sql, assumption, suggestions, key=f"s{len(messages) + 1}")
    messages += [
        {"role": "user", "content": question},
        {
            "role": "assistant",
            "content": reply,
            "sql": sql,
            "assumption": assumption,
            "suggestions": suggestions,
        },
    ]


st.title("Chat with your data")
try:
    stats = load_stats()
except FileNotFoundError as e:
    st.error(str(e))
    st.stop()
show_overview(stats)
show_chat(stats)
