"""Dashboard: a chat box that opens with a short intro to the data.

Questions go through the same pipeline and logging as the command-line app (app.ask).
Run from the project folder with:

    streamlit run src/dashboard.py
"""

import logging
import uuid

import altair as alt
import pandas as pd
import streamlit as st

from config import Config
from db import connect_readonly
from logger import setup_logging

st.set_page_config(page_title="Olist data chat", layout="wide")

ROBOT = (
    Config.BASE_DIR / "assets" / "robot.svg"
)  # kept; swap in for ASSISTANT_AVATAR to use it
CAPYBARA = Config.BASE_DIR / "assets" / "capybara.svg"
ASSISTANT_AVATAR = CAPYBARA  # the assistant's picture in the chat
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


@st.cache_resource
def start_logging() -> bool:
    """Open this server's log files once (Streamlit reruns the script on every click).

    If the log folders can't be created the app still runs, just without logs.
    """
    try:
        setup_logging()
    except OSError:
        log.exception("Logging could not be started; continuing without log files")
        return False
    log.info("dashboard start: db=%s model=%s", Config.DB_PATH, Config.MODEL)
    return True


def get_answer(question: str, source: str = "typed", follow_up_of: str = "") -> dict:
    """Run one question through app.ask(), which answers and logs it.

    source is where the question came from: "typed", "example" (a welcome button)
    or "follow_up" (a suggestion under the answer whose question_id is
    follow_up_of). Returns the reply as the fields of a chat message.
    """
    try:
        app, system = load_pipeline()
        # A new read-only connection per question: SQLite connections can't be
        # shared between the threads Streamlit runs the page in.
        conn = connect_readonly(Config.DB_PATH)
    except RuntimeError as e:  # llm.py: the API key is not set
        log.warning("Model not available: %s", e)
        return _reply("", f"The model isn't available: {e}")
    except Exception:
        log.exception("Could not start answering: %s", question)
        return _reply("", "Something went wrong answering that. Please try again.")

    try:
        result, question_id = app.ask(
            conn,
            system,
            question,
            source=source,
            follow_up_of=follow_up_of,
            session_id=st.session_state.get("session_id", ""),
        )
    finally:
        conn.close()
    return _reply(
        question_id, result.message, result.sql, result.assumption, result.suggestions
    )


def _reply(
    question_id: str,
    content: str,
    sql: str = "",
    assumption: str = "",
    suggestions: list[str] | None = None,
) -> dict:
    """The fields of an assistant chat message."""
    return {
        "question_id": question_id,
        "content": content,
        "sql": sql,
        "assumption": assumption,
        "suggestions": suggestions or [],
    }


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
    return f"""**Hi, I'm Capi! Ask me about this data and I'll walk you through it.**

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
    """Capybara for the assistant, green person for the user.

    Falls back to Streamlit's default avatar if an image file is missing.
    """
    image = ASSISTANT_AVATAR if role == "assistant" else USER
    return str(image) if image.exists() else None


def ask_this(question: str, source: str, follow_up_of: str = "") -> None:
    """Button callback: queue the question; show_chat picks it up on the rerun.

    source and follow_up_of travel with it, so the log shows where it came from.
    """
    st.session_state.pending = {
        "question": question,
        "source": source,
        "follow_up_of": follow_up_of,
    }


WALKING_CAPYBARA = (
    Config.BASE_DIR / "assets" / "walking_capybara.svg"
)  # shown while thinking


def show_body(
    content: str,
    sql: str = "",
    assumption: str = "",
    suggestions: tuple | list = (),
    key: str = "",
    button_source: str = "follow_up",
    follow_up_of: str = "",
) -> None:
    """A message's text, how the question was read, the SQL (closed at first)
    and suggested questions as buttons. key keeps the buttons' ids unique;
    button_source and follow_up_of say, for the log, where a clicked one came from."""
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
                args=(suggestion, button_source, follow_up_of),
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
    st.session_state.setdefault("session_id", uuid.uuid4().hex[:8])

    # A clicked suggestion is queued in session state and handled like typed input.
    typed = st.chat_input("Ask a question about the data...")
    pending = st.session_state.pop("pending", None)
    if pending:
        question, source, follow_up_of = (
            pending["question"],
            pending["source"],
            pending["follow_up_of"],
        )
    else:
        question, source, follow_up_of = typed, "typed", ""

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
                # Buttons under the welcome message are examples; under an answer
                # they are follow-ups to that answer.
                button_source="follow_up" if message.get("question_id") else "example",
                follow_up_of=message.get("question_id", ""),
            )

    if not question:
        return
    # Show the question straight away, then the reply once it is ready.
    with st.chat_message("user", avatar=avatar("user")):
        st.markdown(question)
    with st.chat_message("assistant", avatar=avatar("assistant")):
        waiting = st.empty()
        with waiting.container(horizontal=True, vertical_alignment="center"):
            st.image(
                str(WALKING_CAPYBARA), width=96
            )  # the capybara walks until the answer is ready
            st.caption("Walking...")
        reply = get_answer(question, source, follow_up_of)
        waiting.empty()
        # The reply will sit at index len(messages) + 1, so its button ids match
        # the ones the history loop uses on the next run.
        show_body(
            reply["content"],
            reply["sql"],
            reply["assumption"],
            reply["suggestions"],
            key=f"s{len(messages) + 1}",
            follow_up_of=reply["question_id"],
        )
    messages += [
        {"role": "user", "content": question},
        {"role": "assistant", **reply},
    ]
    st.html(
        f"<script>{SCROLL_TO_LATEST_JS.format(count=len(messages))}</script>",
        unsafe_allow_javascript=True,
    )


# Streamlit scrolls chat pages down to the newest message, which on first load skips
# past the title and the start of the welcome message. Its scrolling is an animation plus
# a "follow the bottom" flag that it only clears when it sees a scroll event on a page
# whose size has not changed. So for the first seconds the page ignores programmatic
# scrolling (nothing moves), harmless scroll events are sent so that flag settles to
# "free", and scrolling is handed back once Streamlit has been quiet for a moment or the
# visitor scrolls, clicks or types.
KEEP_AT_TOP_JS = """
(() => {
  const page = document.querySelector('main');
  if (!page) return;
  let blocking = true;
  let attempts = 0;  // how many times Streamlit tried to scroll the page
  let lastAttempt = 0;
  const top = Object.getOwnPropertyDescriptor(Element.prototype, 'scrollTop');
  const intoView = Element.prototype.scrollIntoView;
  const swallow = () => { attempts += 1; lastAttempt = Date.now(); };
  // The visitor's own scrolling (wheel, touch, keys) is native and never goes through
  // these, so it is never blocked.
  Object.defineProperty(page, 'scrollTop', {
    configurable: true,
    get() { return top.get.call(this); },
    set(value) { if (blocking) swallow(); else top.set.call(this, value); },
  });
  ['scrollTo', 'scroll', 'scrollBy'].forEach((name) => {
    page[name] = (...args) => {
      if (blocking) swallow(); else Element.prototype[name].apply(page, args);
    };
  });
  Element.prototype.scrollIntoView = function (...args) {
    if (blocking && page.contains(this)) { swallow(); return; }
    return intoView.apply(this, args);
  };
  let watcher, cap;
  const release = () => {
    blocking = false;
    clearInterval(watcher);
    clearTimeout(cap);
    delete page.scrollTop;
    ['scrollTo', 'scroll', 'scrollBy'].forEach((name) => delete page[name]);
    Element.prototype.scrollIntoView = intoView;
  };
  watcher = setInterval(() => {
    page.dispatchEvent(new Event('scroll'));  // lets Streamlit settle into "free"
    // Streamlit has tried to scroll and then stopped: its animation is over.
    if (attempts > 0 && Date.now() - lastAttempt > 1500) release();
  }, 120);
  ['wheel', 'touchstart', 'keydown', 'mousedown'].forEach((name) =>
    window.addEventListener(name, release, { once: true, passive: true })
  );
  cap = setTimeout(release, 15000);
})();
"""

# Streamlit's own scrolling is held back at first (above), so when a question is asked
# the new answer is scrolled into view explicitly. The number in the comment makes each
# script different, so that it runs again every time.
SCROLL_TO_LATEST_JS = """
/* {count} */
setTimeout(() => {{
  const page = document.querySelector('main');
  if (page) page.scrollTo({{ top: page.scrollHeight, behavior: 'smooth' }});
}}, 80);
"""


def keep_page_at_top() -> None:
    """On the first load of a session, stop the page from sliding down to the bottom."""
    if "scroll_reset" not in st.session_state:
        st.session_state.scroll_reset = True
        st.html(f"<script>{KEEP_AT_TOP_JS}</script>", unsafe_allow_javascript=True)


keep_page_at_top()  # first, so it is in place before the page can scroll
start_logging()
st.title("Talk with your data: meet Capi")
try:
    stats = load_stats()
except FileNotFoundError as e:
    st.error(str(e))
    st.stop()
show_overview(stats)
show_chat(stats)
