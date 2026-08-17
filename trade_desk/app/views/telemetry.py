"""LLM usage / cost telemetry page."""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from modules.db import llm_calls_summary


def render_telemetry_page(st_obj):
    st_obj.title("💰 LLM Usage")
    st_obj.caption("Every Haiku call this app has made, and what it cost.")

    stats = llm_calls_summary()

    if stats["total_calls"] == 0:
        st_obj.info("No LLM calls logged yet. Trigger an AI rationale (Haiku backend) on the Analyze page to see data here.")
        return

    c1, c2, c3, c4 = st_obj.columns(4)
    c1.metric("Total cost", f"${stats['total_cost']:.4f}")
    c2.metric("Calls", stats["total_calls"])
    c3.metric("Input tokens", f"{stats['total_in_tokens']:,}")
    c4.metric("Output tokens", f"{stats['total_out_tokens']:,}")

    st_obj.divider()

    col_a, col_b = st_obj.columns(2)
    with col_a:
        st_obj.subheader("By source")
        st_obj.dataframe(
            [{"source": r["source"], "calls": r["n"], "cost_usd": round(r["cost"], 6),
              "in_tok": r["in_tok"], "out_tok": r["out_tok"]} for r in stats["by_source"]],
            hide_index=True, use_container_width=True,
        )
        st_obj.caption("haiku_cli = subscription quota, not billed $ — tokens shown but cost is $0 by design.")
    with col_b:
        st_obj.subheader("By day")
        if stats["by_day"]:
            st_obj.bar_chart({r["day"]: r["cost"] for r in stats["by_day"]})

    st_obj.divider()
    st_obj.subheader("Recent calls")
    st_obj.dataframe(
        [{"ticker": r["ticker"], "backend": r["backend"], "source": r["source"],
          "in_tok": r["in_tokens"], "out_tok": r["out_tokens"], "cost_usd": round(r["cost_usd"], 6)}
         for r in stats["recent"]],
        hide_index=True, use_container_width=True,
    )
