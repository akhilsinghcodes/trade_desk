"""ELI5 (Explain Like I'm 5) page."""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from modules.eli5 import CATEGORIES, get_by_category, search_terms as eli5_search


category_colors = {
    "Technical": "#4fc3f7",
    "Fundamental": "#00c853",
    "Valuation": "#ff9100",
    "Market Intelligence": "#b39ddb",
    "Risk": "#ff1744",
    "Verdict": "#ffd740",
}


def render_eli5_page(st_obj):
    """Render ELI5 page."""
    st_obj.title("📚 Explain Like I'm 5")
    st_obj.caption("Every term, chart, and signal in plain English. No jargon.")

    search_q = st_obj.text_input("🔍 Search a term...", placeholder="e.g. RSI, short squeeze, P/E ratio")
    st_obj.divider()

    def render_term_card(term, data):
        cat = data.get("category", "")
        cat_color = category_colors.get(cat, "#888")
        with st_obj.expander(f"{data.get('emoji','📌')} **{term}** — {data.get('full_name', '')}"):
            st_obj.markdown(
                f'<span style="background:{cat_color}22;color:{cat_color};padding:2px 8px;border-radius:12px;font-size:0.75rem;font-weight:600">{cat}</span>',
                unsafe_allow_html=True
            )
            st_obj.markdown(f"### 🧒 Simple version\n{data['eli5']}")
            st_obj.markdown(f"**What it really means:** {data['what_it_means']}")
            col_g, col_b = st_obj.columns(2)
            col_g.success(f"✅ Good: {data['good_sign']}")
            col_b.error(f"⚠️ Watch out: {data['bad_sign']}")

    if search_q.strip():
        results = eli5_search(search_q.strip())
        if results:
            st_obj.caption(f"{len(results)} result(s) for '{search_q}'")
            for term, data in results:
                render_term_card(term, data)
        else:
            st_obj.info("No matching terms. Try a shorter keyword.")
    else:
        for cat in CATEGORIES:
            entries = get_by_category(cat)
            if not entries:
                continue
            cat_color = category_colors.get(cat, "#888")
            st_obj.markdown(
                f'<h3 style="color:{cat_color};margin-top:24px">{cat}</h3>',
                unsafe_allow_html=True
            )
            for term, data in entries:
                render_term_card(term, data)

    st_obj.stop()
