"""Exercise 5 — Streamlit approval UI for the HITL PR review agent.

Run with:
    streamlit run app.py

Goal: wrap the LangGraph built in exercises 1–4 in a web UI that adapts to
the confidence bucket of each PR.

Routing thresholds (common/schemas.py):
    > 72%        auto_approve     UI shows a success card; reviewer does nothing
    58 – 72%     human_approval   UI shows Approve / Reject / Edit buttons
    <  58%       escalate         UI shows a question form for the reviewer
"""

from __future__ import annotations

import asyncio
import sys
import uuid
from pathlib import Path

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent))

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

import streamlit as st
from dotenv import load_dotenv
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.types import Command

from common.db import db_conn, db_path
from exercises.exercise_4_audit import build_graph


load_dotenv()


# ─── Session state ─────────────────────────────────────────────────────────
if "thread_id" not in st.session_state:
    st.session_state.thread_id = None
if "pr_url" not in st.session_state:
    st.session_state.pr_url = ""
if "interrupt_payload" not in st.session_state:
    st.session_state.interrupt_payload = None
if "final" not in st.session_state:
    st.session_state.final = None


# ─── Page setup ────────────────────────────────────────────────────────────
st.set_page_config(page_title="HITL PR Review Agent", page_icon="🤖", layout="wide")
st.title("🤖 HITL Pull Request Review Agent")
st.caption("Human-in-the-loop PR review system powered by LangGraph & SQLite Audit Trail")


# ─── Sidebar — recent sessions ─────────────────────────────────────────────
with st.sidebar:
    st.header("📋 Recent Sessions")

    async def fetch_recent_threads():
        async with db_conn() as conn:
            async with conn.execute(
                """
                SELECT thread_id,
                       pr_url,
                       MIN(timestamp)        AS started,
                       MAX(timestamp)        AS last_event,
                       MAX(risk_level)       AS worst_risk,
                       COUNT(*)              AS events
                  FROM audit_events
                 GROUP BY thread_id, pr_url
                 ORDER BY MAX(timestamp) DESC
                 LIMIT 15
                """
            ) as cur:
                rows = await cur.fetchall()
        return [dict(r) for r in rows]

    try:
        threads = asyncio.run(fetch_recent_threads())
        if threads:
            for t in threads:
                risk = t.get("worst_risk", "low")
                badge = "🔴" if risk == "high" else ("🟡" if risk == "med" else "🟢")
                btn_label = f"{badge} {t['thread_id'][:8]}… ({t['events']} ev)"
                if st.button(btn_label, key=f"hist_{t['thread_id']}", use_container_width=True):
                    st.session_state.thread_id = t["thread_id"]
                    st.session_state.pr_url = t["pr_url"]
                    st.session_state.interrupt_payload = None
                    st.session_state.final = None
                    st.rerun()
                st.caption(f"{t['pr_url'].split('/')[-1] if '/' in t['pr_url'] else t['pr_url']} · {t['last_event'][11:19]}")
        else:
            st.caption("No previous sessions recorded yet.")
    except Exception as e:
        st.caption(f"No history available ({e})")


# ─── Top form — start a new review ─────────────────────────────────────────
with st.form("start"):
    st.markdown("#### Analyze Pull Request")
    pr_url = st.text_input(
        "GitHub PR URL",
        value=st.session_state.pr_url,
        placeholder="https://github.com/VinUni-AI20k/PR-Demo/pull/1",
    )
    col_submit, col_hint = st.columns([1, 4])
    with col_submit:
        submitted = st.form_submit_button("Run review", type="primary", use_container_width=True)
    with col_hint:
        st.caption("Try demo PR #1 (priority field) or PR #2 (user login & sync).")


# ─── Renderers per interrupt kind ──────────────────────────────────────────
def render_approval_card(payload: dict) -> dict | None:
    """58–72% bucket: show the LLM review + 3 buttons. Return resume dict or None."""
    conf = payload["confidence"]
    st.info(f"🟡 **Approval Requested — Confidence {conf:.0%}** (Medium Risk)")
    st.write(f"**Reasoning:** {payload['confidence_reasoning']}")
    st.markdown(f"**Summary:** {payload['summary']}")

    if payload.get("comments"):
        st.markdown("##### Proposed Review Comments:")
        for c in payload.get("comments", []):
            st.markdown(f"- **[{c['severity'].upper()}]** `{c['file']}:{c.get('line') or '?'}` — {c['body']}")

    with st.expander("🔍 View PR Unified Diff", expanded=False):
        st.code(payload.get("diff_preview", ""), language="diff")

    feedback = st.text_input("Feedback or reviewer note (optional)", key="approval_feedback")
    col1, col2, col3 = st.columns(3)
    if col1.button("✅ Approve", type="primary", use_container_width=True):
        return {"choice": "approve", "feedback": feedback}
    if col2.button("❌ Reject", use_container_width=True):
        return {"choice": "reject", "feedback": feedback}
    if col3.button("✏️ Edit & Request Changes", use_container_width=True):
        return {"choice": "edit", "feedback": feedback}
    return None


def render_escalation_card(payload: dict) -> dict | None:
    """< 58% bucket: show risk factors + question form. Return {question: answer} or None."""
    conf = payload["confidence"]
    st.error(f"🔴 **Strong Escalation — Confidence {conf:.0%}** (High Risk)")
    st.write(f"**Reasoning:** {payload['confidence_reasoning']}")
    if payload.get("risk_factors"):
        st.warning("**Identified Risk Factors:**\n" + "\n".join(f"- {r}" for r in payload["risk_factors"]))
    st.markdown(f"**Summary:** {payload['summary']}")

    with st.form("escalation_form"):
        st.markdown("##### ❓ Reviewer Clarifications Required:")
        answers: dict[str, str] = {}
        questions = payload.get("questions", [])
        for idx, q in enumerate(questions):
            answers[q] = st.text_input(q, key=f"q_{idx}")
        submitted_answers = st.form_submit_button("Submit answers & Refine Review", type="primary")
        if submitted_answers:
            return answers
    return None


# ─── Drive the graph ───────────────────────────────────────────────────────
async def run_graph(pr_url: str, thread_id: str, resume_value=None):
    """Invoke the graph once. Returns the final result or {'__interrupt__': ...}."""
    async with AsyncSqliteSaver.from_conn_string(db_path()) as cp:
        await cp.setup()
        app = build_graph(cp)
        cfg = {"configurable": {"thread_id": thread_id}}

        if resume_value is None:
            result = await app.ainvoke({"pr_url": pr_url, "thread_id": thread_id}, cfg)
        else:
            result = await app.ainvoke(Command(resume=resume_value), cfg)
        return result


# ─── Main flow ─────────────────────────────────────────────────────────────
if submitted and pr_url:
    st.session_state.pr_url = pr_url
    st.session_state.thread_id = str(uuid.uuid4())
    st.session_state.interrupt_payload = None
    st.session_state.final = None

    with st.spinner("Fetching PR & analyzing diff with LLM..."):
        result = asyncio.run(run_graph(pr_url, st.session_state.thread_id))

    if "__interrupt__" in result:
        st.session_state.interrupt_payload = result["__interrupt__"][0].value
    else:
        st.session_state.final = result
    st.rerun()

# Render the current interrupt card, if any
payload = st.session_state.interrupt_payload
if payload is not None:
    kind = payload["kind"]
    answer = render_approval_card(payload) if kind == "approval_request" else render_escalation_card(payload)
    if answer is not None:
        with st.spinner("Resuming agent with your decision..."):
            result = asyncio.run(run_graph(
                st.session_state.pr_url, st.session_state.thread_id, resume_value=answer,
            ))
        if "__interrupt__" in result:
            st.session_state.interrupt_payload = result["__interrupt__"][0].value
        else:
            st.session_state.interrupt_payload = None
            st.session_state.final = result
        st.rerun()

# Render final state, if reached
if st.session_state.final is not None:
    final = st.session_state.final
    action = final.get("final_action", "?")
    st.divider()

    if action.startswith("auto") or action.startswith("committed"):
        st.success(f"🎉 **Review Posted Successfully!** (`final_action = {action}`)")
        st.markdown(f"[🔗 View discussion on GitHub]({st.session_state.pr_url})")
    elif action == "rejected":
        st.warning(f"🛑 **Review Rejected** (`final_action = {action}`). No comment posted to GitHub.")
    else:
        st.info(f"ℹ️ `final_action = {action}`")

    if "analysis" in final:
        a = final["analysis"]
        st.markdown(f"**Final Confidence:** `{a.confidence:.0%}`")
        st.markdown(f"**Final Summary:** {a.summary}")

    st.caption(
        f"Thread ID: `{st.session_state.thread_id}` · "
        f"Replay command: `python -m audit.replay --thread {st.session_state.thread_id}`"
    )
