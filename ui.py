"""
NiceGUI interactive UI for the AI Customer Support triage system.
Connects directly to the same triage service used by the API and CLI.

Run with: python ui.py
"""

import json
from nicegui import ui
from src.db import MockDB
from src.config import AppContext
from src.triage_service import run_triage_stream_events
from src.logger import get_logger

logger = get_logger(__name__)

db_instance = MockDB()

CATEGORY_COLORS = {
    "refund": "#f97316",          # orange
    "technical_support": "#3b82f6",  # blue
    "general_query": "#22c55e",   # green
    "unknown": "#ef4444",         # red
}


def parse_sse_event(raw: str) -> dict | None:
    """Strip the SSE 'data: ' prefix and parse the JSON payload."""
    line = raw.strip()
    if line.startswith("data: "):
        try:
            return json.loads(line[6:])
        except json.JSONDecodeError:
            return None
    return None


@ui.page("/")
def index():
    ui.query("body").style("background: #0f172a")

    with ui.column().classes("w-full max-w-2xl mx-auto px-4 py-10 gap-6"):

        # ── Header ──────────────────────────────────────────────────────────
        with ui.column().classes("gap-1"):
            ui.label("AI Customer Support").classes(
                "text-3xl font-bold text-white"
            )
            ui.label("Multi-agent triage powered by Pydantic AI").classes(
                "text-slate-400 text-sm"
            )

        # ── Input card ──────────────────────────────────────────────────────
        with ui.card().classes("w-full bg-slate-800 border border-slate-700 rounded-xl p-5 gap-4"):
            with ui.row().classes("w-full gap-3 items-start"):
                email_input = (
                    ui.input(placeholder="customer@example.com")
                    .props('outlined dense label="Customer Email"')
                    .classes("flex-1 text-white")
                )
                query_input = (
                    ui.textarea(placeholder="e.g. I want a refund for order #123")
                    .props('outlined dense rows="1" label="Customer Query"')
                    .classes("flex-1 text-white")
                )
            submit_btn = (
                ui.button("Run Agent", icon="play_arrow")
                .props("unelevated")
                .classes("w-full bg-indigo-600 hover:bg-indigo-500 text-white font-semibold py-2 rounded-lg")
            )

        # ── Status / progress area ───────────────────────────────────────────
        status_label = (
            ui.label("")
            .classes("text-slate-400 text-sm italic px-1 hidden")
        )

        # ── Live reply box ──────────────────────────────────────────────────
        reply_card = (
            ui.card()
            .classes("w-full bg-slate-800 border border-slate-700 rounded-xl p-5 hidden")
        )
        with reply_card:
            ui.label("Customer Reply").classes("text-slate-400 text-xs uppercase tracking-widest mb-2")
            reply_label = ui.label("").classes("text-white text-base leading-relaxed")

        # ── Final result card ───────────────────────────────────────────────
        result_card = (
            ui.card()
            .classes("w-full bg-slate-800 border border-slate-700 rounded-xl p-5 hidden")
        )
        with result_card:
            ui.label("Triage Result").classes("text-slate-400 text-xs uppercase tracking-widest mb-3")
            with ui.grid(columns=2).classes("w-full gap-y-2 gap-x-4"):
                category_key   = ui.label("Category").classes("text-slate-500 text-sm")
                category_val   = ui.badge("").classes("text-sm font-mono self-center")
                approval_key   = ui.label("Human Approval").classes("text-slate-500 text-sm")
                approval_val   = ui.badge("").classes("text-sm self-center")
                order_key      = ui.label("Order ID").classes("text-slate-500 text-sm")
                order_val      = ui.label("").classes("text-white text-sm")

            ui.separator().classes("my-3 border-slate-700")
            ui.label("Suggested Action").classes("text-slate-400 text-xs uppercase tracking-widest mb-1")
            action_label = ui.label("").classes("text-slate-200 text-sm leading-relaxed")

        # ── Event log ───────────────────────────────────────────────────────
        with ui.expansion("Raw event log", icon="terminal").classes(
            "w-full text-slate-400 bg-slate-900 rounded-xl border border-slate-700"
        ):
            log = ui.log(max_lines=100).classes(
                "w-full h-48 font-mono text-xs text-green-400 bg-transparent"
            )

        # ── Submit handler ──────────────────────────────────────────────────
        async def on_submit():
            email = email_input.value.strip()
            query = query_input.value.strip()

            if not email or not query:
                ui.notify("Please fill in both email and query.", type="warning")
                return

            logger.info(f"Triage request received | email={email} | query={query}")

            # Reset UI state
            submit_btn.props("loading")
            reply_card.classes(add="hidden")
            reply_label.set_text("")
            result_card.classes(add="hidden")
            status_label.classes(remove="hidden")
            status_label.classes(remove="text-green-400")
            status_label.classes("text-slate-400")
            status_label.set_text("⚙ Starting agent...")
            log.clear()

            ctx = AppContext(db=db_instance, user_email=email)

            async for raw_event in run_triage_stream_events(ctx, query):
                log.push(raw_event.strip())
                payload = parse_sse_event(raw_event)
                if payload is None:
                    continue

                if "status" in payload:
                    # Tool callback events -- update status label so user sees live progress
                    status_label.set_text(f"⚙ {payload['status']}")

                elif "customer_reply" in payload:
                    # Partial token events -- silently accumulate, don't show yet
                    pass

                elif "final" in payload:
                    final = payload["final"]

                    # Now reveal the reply card with the completed text
                    reply_label.set_text(final.get("customer_reply", ""))
                    reply_card.classes(remove="hidden")

                    # Populate result card
                    category = final.get("category", "unknown")
                    color = CATEGORY_COLORS.get(category, "#94a3b8")
                    category_val.set_text(category.replace("_", " ").title())
                    category_val.style(f"background-color: {color}; color: white;")

                    needs_approval = final.get("requires_human_approval", False)
                    approval_val.set_text("Required" if needs_approval else "Not Required")
                    approval_val.style(
                        "background-color: #f97316; color: white;"
                        if needs_approval
                        else "background-color: #22c55e; color: white;"
                    )

                    order_val.set_text(final.get("order_id") or "—")
                    action_label.set_text(final.get("suggested_action", ""))
                    result_card.classes(remove="hidden")

                    # Update status to done
                    status_label.classes(remove="text-slate-400")
                    status_label.classes("text-green-400")
                    status_label.set_text("✓ Done")

            submit_btn.props(remove="loading")

        submit_btn.on_click(on_submit)


ui.run(
    title="AI Customer Support",
    port=8080,
    reload=False,
    dark=True,
)
