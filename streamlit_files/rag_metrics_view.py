"""RAG evaluation metrics dashboard for Streamlit."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from RAG_eval import (
    INVERTED_METRIC_LABELS,
    METRIC_LABELS,
    METRIC_REGISTRY,
    EvalConfig,
    run_evaluation,
)
from rate_limits import (
    DAILY_TOKEN_LIMIT,
    RAG_METRICS_SESSION_LIMIT,
    SESSION_TOKEN_LIMIT,
    DailyTokenLimiter,
    RagMetricsSessionLimiter,
    SessionTokenLimiter,
    estimate_eval_run_tokens,
)
from rag_eval_goldens import GOLDENS

EVAL_RESULTS_PATH = PROJECT_ROOT / ".deepeval" / ".latest_test_run.json"

METRIC_COLORS = {
    "Contextual Recall": "#6366f1",
    "Contextual Precision": "#8b5cf6",
    "Contextual Relevancy": "#38bdf8",
    "Faithfulness": "#10b981",
    "Answer Relevancy": "#f59e0b",
    "Hallucination": "#f43f5e",
    "Non-Advice": "#14b8a6",
}

CHART_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color="#e2e8f0"),
    margin=dict(l=20, r=20, t=40, b=20),
)


def load_eval_results(path: Path = EVAL_RESULTS_PATH) -> dict | None:
    if not path.exists():
        return None
    try:
        with path.open(encoding="utf-8") as handle:
            return json.load(handle)
    except (json.JSONDecodeError, OSError):
        return None


def _parse_test_cases(data: dict) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    test_cases = data.get("testRunData", {}).get("testCases", [])
    rows: list[dict] = []
    metric_rows: list[dict] = []

    for index, case in enumerate(test_cases):
        question = case.get("input", f"Test case {index + 1}")
        short_question = question if len(question) <= 60 else f"{question[:57]}..."
        case_success = case.get("success", False)

        for metric in case.get("metricsData", []):
            name = metric.get("name", "Unknown")
            score = float(metric.get("score", 0.0))
            passed = bool(metric.get("success", False))

            rows.append(
                {
                    "case_id": index + 1,
                    "question": short_question,
                    "full_question": question,
                    "metric": name,
                    "score": score,
                    "passed": passed,
                    "threshold": metric.get("threshold", 0.5),
                    "reason": metric.get("reason", ""),
                }
            )
            metric_rows.append({"metric": name, "score": score, "passed": passed})

        rows.append(
            {
                "case_id": index + 1,
                "question": short_question,
                "full_question": question,
                "metric": "__case__",
                "score": 1.0 if case_success else 0.0,
                "passed": case_success,
                "threshold": 0.5,
                "reason": "",
            }
        )

    details_df = pd.DataFrame(rows)
    if details_df.empty:
        summary_df = pd.DataFrame(
            columns=[
                "metric",
                "avg_score",
                "pass_rate",
                "passes",
                "fails",
                "inverted",
                "display_score",
            ]
        )
        return details_df, summary_df, {}

    metric_df = details_df[details_df["metric"] != "__case__"]
    summary_df = (
        metric_df.groupby("metric", as_index=False)
        .agg(
            avg_score=("score", "mean"),
            pass_rate=("passed", "mean"),
            passes=("passed", "sum"),
            fails=("passed", lambda s: int((~s).sum())),
        )
        .sort_values("metric")
    )
    # Hallucination scores run the other way (0 is a perfect run). Charts and
    # the headline average read "higher is better", so align it here and keep
    # the raw value for the detail view.
    summary_df["inverted"] = summary_df["metric"].isin(INVERTED_METRIC_LABELS)
    summary_df["display_score"] = summary_df["avg_score"].where(
        ~summary_df["inverted"], 1 - summary_df["avg_score"]
    )

    run_data = data.get("testRunData", {})
    meta = {
        "total_cases": len(test_cases),
        "passed_cases": int(sum(1 for case in test_cases if case.get("success"))),
        "failed_cases": int(sum(1 for case in test_cases if not case.get("success"))),
        "run_duration": run_data.get("runDuration"),
        "evaluation_cost": run_data.get("evaluationCost", 0.0),
    }
    return details_df, summary_df, meta


def _score_gauge(
    score: float,
    title: str,
    color: str,
    threshold: float = 0.5,
    inverted: bool = False,
) -> go.Figure:
    # An inverted metric passes below its threshold; the gauge plots the
    # aligned score, so the pass mark sits at the mirrored position.
    mark = (1 - threshold) if inverted else threshold
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=round(score * 100, 1),
            number={"suffix": "%", "font": {"size": 28}},
            title={"text": f"{title} (inv.)" if inverted else title, "font": {"size": 14}},
            gauge={
                "axis": {"range": [0, 100], "tickwidth": 1},
                "bar": {"color": color},
                "bgcolor": "rgba(30,41,59,0.6)",
                "borderwidth": 0,
                "steps": [
                    {"range": [0, 50], "color": "rgba(239,68,68,0.25)"},
                    {"range": [50, 75], "color": "rgba(245,158,11,0.25)"},
                    {"range": [75, 100], "color": "rgba(16,185,129,0.25)"},
                ],
                "threshold": {
                    "line": {"color": "#f8fafc", "width": 2},
                    "thickness": 0.8,
                    "value": mark * 100,
                },
            },
        )
    )
    fig.update_layout(height=220, **CHART_LAYOUT)
    return fig


def _metric_bar_chart(summary_df: pd.DataFrame, threshold: float = 0.5) -> go.Figure:
    chart_df = summary_df.copy()
    chart_df["color"] = chart_df["metric"].map(METRIC_COLORS).fillna("#94a3b8")
    labels = [
        f"{name} (inv.)" if inverted else name
        for name, inverted in zip(chart_df["metric"], chart_df["inverted"])
    ]
    fig = go.Figure(
        go.Bar(
            x=labels,
            y=chart_df["display_score"],
            marker_color=chart_df["color"],
            text=[f"{value:.0%}" for value in chart_df["display_score"]],
            textposition="outside",
        )
    )
    fig.add_hline(
        y=threshold,
        line_dash="dash",
        line_color="#94a3b8",
        annotation_text="Threshold",
    )
    fig.update_layout(
        title="Average Metric Scores",
        yaxis=dict(title="Score", range=[0, 1.05], tickformat=".0%"),
        xaxis_title="",
        height=360,
        **CHART_LAYOUT,
    )
    return fig


def _radar_chart(summary_df: pd.DataFrame) -> go.Figure:
    metrics = [
        f"{name} (inv.)" if inverted else name
        for name, inverted in zip(summary_df["metric"], summary_df["inverted"])
    ]
    scores = summary_df["display_score"].tolist()
    if not metrics:
        return go.Figure()

    metrics_closed = metrics + [metrics[0]]
    scores_closed = scores + [scores[0]]

    fig = go.Figure(
        go.Scatterpolar(
            r=scores_closed,
            theta=metrics_closed,
            fill="toself",
            fillcolor="rgba(99,102,241,0.35)",
            line=dict(color="#818cf8", width=2),
            name="RAG pipeline",
        )
    )
    fig.update_layout(
        title="Metric Balance",
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 1], tickformat=".0%"),
            bgcolor="rgba(15,23,42,0.4)",
        ),
        height=380,
        showlegend=False,
        **CHART_LAYOUT,
    )
    return fig


def _heatmap(details_df: pd.DataFrame) -> go.Figure:
    metric_df = details_df[details_df["metric"] != "__case__"]
    if metric_df.empty:
        return go.Figure()

    pivot = metric_df.pivot_table(
        index="question",
        columns="metric",
        values="score",
        aggfunc="mean",
    )
    fig = px.imshow(
        pivot,
        aspect="auto",
        color_continuous_scale=["#ef4444", "#f59e0b", "#10b981"],
        zmin=0,
        zmax=1,
        labels=dict(color="Score"),
    )
    fig.update_layout(
        title="Per-Question Metric Heatmap",
        height=max(280, 48 * len(pivot.index) + 120),
        **CHART_LAYOUT,
    )
    return fig


def _pass_fail_donut(meta: dict) -> go.Figure:
    labels = ["Passed", "Failed"]
    values = [meta.get("passed_cases", 0), meta.get("failed_cases", 0)]
    colors = ["#10b981", "#ef4444"]

    fig = go.Figure(
        go.Pie(
            labels=labels,
            values=values,
            hole=0.55,
            marker=dict(colors=colors),
            textinfo="label+value",
        )
    )
    fig.update_layout(title="Test Case Outcomes", height=320, **CHART_LAYOUT)
    return fig


def _render_run_controls() -> None:
    total_goldens = len(GOLDENS)
    metric_options = {
        METRIC_LABELS.get(key, key): key for key in METRIC_REGISTRY
    }
    rag_run_limiter = RagMetricsSessionLimiter()
    session_limiter = SessionTokenLimiter()
    daily_limiter = DailyTokenLimiter()
    runs_remaining = rag_run_limiter.runs_remaining(st.session_state)
    session_tokens_remaining = session_limiter.remaining(st.session_state)

    st.caption(
        f"Session tokens remaining: **{session_tokens_remaining:,}** / {SESSION_TOKEN_LIMIT:,} · "
        f"RAG runs remaining: **{runs_remaining}** / {RAG_METRICS_SESSION_LIMIT} · "
        f"Global daily tokens remaining: **{daily_limiter.remaining():,}** / {DAILY_TOKEN_LIMIT:,}"
    )

    with st.expander("Run evaluation", expanded=False):
        st.caption(
            "Configure and launch a DeepEval run. Results are saved to "
            f"`{EVAL_RESULTS_PATH.relative_to(PROJECT_ROOT)}`."
        )

        left, right = st.columns(2)
        with left:
            offset = st.number_input(
                "Skip first N goldens",
                min_value=0,
                max_value=max(total_goldens - 1, 0),
                value=0,
                step=1,
            )
            limit = st.number_input(
                "Number of goldens to evaluate",
                min_value=1,
                max_value=total_goldens,
                value=min(3, total_goldens),
                step=1,
            )
            threshold = st.slider(
                "Pass threshold",
                min_value=0.0,
                max_value=1.0,
                value=0.5,
                step=0.05,
            )

        with right:
            selected_labels = st.multiselect(
                "Metrics",
                options=list(metric_options),
                default=list(metric_options),
            )
            judge_model = st.text_input(
                "OpenRouter judge model",
                value="tencent/hy3:free",
            )
            run_label = st.text_input(
                "Run label (optional)",
                value="",
                placeholder="e.g. baseline-v1",
            )

        selected_metrics = [metric_options[label] for label in selected_labels]
        selected_count = min(limit, max(total_goldens - offset, 0))

        st.info(
            f"Will evaluate **{selected_count}** golden(s) "
            f"(offset {offset}, limit {limit}) using "
            f"**{len(selected_metrics)}** metric(s)."
        )

        if st.button(
            "Run evaluation",
            type="primary",
            use_container_width=True,
            disabled=runs_remaining <= 0,
        ):
            if not selected_metrics:
                st.error("Select at least one metric.")
                return

            rag_run_check = rag_run_limiter.check_can_run(st.session_state)
            if not rag_run_check.allowed:
                st.error(rag_run_check.message)
                return

            estimated_tokens = estimate_eval_run_tokens(
                selected_count,
                len(selected_metrics),
            )

            session_token_check = session_limiter.check_can_use(
                st.session_state, estimated_tokens
            )
            if not session_token_check.allowed:
                st.error(session_token_check.message)
                return

            daily_check = daily_limiter.check_can_use(estimated_tokens)
            if not daily_check.allowed:
                st.error(daily_check.message)
                return

            config = EvalConfig(
                limit=limit,
                offset=offset,
                threshold=threshold,
                metrics=selected_metrics,
                print_results=False,
                judge_model=judge_model.strip() or "tencent/hy3:free",
                identifier=run_label.strip() or None,
            )

            try:
                with st.spinner(
                    f"Running evaluation on {selected_count} golden(s). "
                    "This can take several minutes..."
                ):
                    run_evaluation(config, session_state=st.session_state)
            except Exception as exc:
                # Only count a run once it actually happened. Charging the
                # attempt up front burned one of the three session runs on
                # failures the user never got results from.
                st.error(f"Evaluation failed: {exc}")
                return
            rag_run_limiter.record_run(st.session_state)

            st.success("Evaluation complete. Refreshing dashboard...")
            st.rerun()


def render_rag_metrics() -> None:
    st.markdown(
        """
        <style>
        .rag-hero {
            background:
                radial-gradient(ellipse 70% 120% at 90% -10%, rgba(129,140,248,0.22), transparent 55%),
                linear-gradient(135deg, #141b38 0%, #1c2150 55%, #0b1120 100%);
            border: 1px solid rgba(129, 140, 248, 0.28);
            border-radius: 16px;
            padding: 1.2rem 1.5rem;
            margin-bottom: 1rem;
        }
        .rag-hero h3 { margin: 0 0 0.3rem 0; color: #e0e7ff; }
        .rag-hero p { margin: 0; color: #a5b4d4; font-size: 0.93rem; }

        /* KPI metric cards */
        [data-testid="stMetric"] {
            background: rgba(22, 33, 58, 0.55);
            border: 1px solid rgba(148, 163, 184, 0.15);
            border-radius: 14px;
            padding: 0.85rem 1rem 0.7rem 1rem;
        }
        [data-testid="stMetricLabel"] p {
            color: #8ea0c0 !important;
            font-size: 0.78rem !important;
            text-transform: uppercase;
            letter-spacing: 0.06em;
        }
        [data-testid="stMetricValue"] { color: #e0e7ff; }
        </style>
        <div class="rag-hero">
            <h3>📊 RAG Evaluation Dashboard</h3>
            <p>DeepEval scores for retrieval quality, faithfulness, and answer relevancy on the statute goldens.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    _render_run_controls()

    col_refresh, col_path = st.columns([1, 3])
    with col_refresh:
        if st.button("Refresh results", use_container_width=True):
            st.rerun()
    with col_path:
        st.caption(f"Reading from `{EVAL_RESULTS_PATH.relative_to(PROJECT_ROOT)}`")

    data = load_eval_results()
    if not data:
        st.info(
            "No evaluation results found yet. Use **Run evaluation** above or run "
            "`python RAG_eval.py --limit 3 --metrics contextual_recall faithfulness` "
            "from the project root, then refresh this tab."
        )
        return

    details_df, summary_df, meta = _parse_test_cases(data)
    if details_df.empty:
        st.warning("Evaluation file exists but contains no test cases.")
        return

    metric_df = details_df[details_df["metric"] != "__case__"]
    overall_pass_rate = meta["passed_cases"] / meta["total_cases"] if meta["total_cases"] else 0.0
    # Average the aligned scores so an inverted metric doesn't drag the
    # headline down for doing well.
    avg_score = summary_df["display_score"].mean() if not summary_df.empty else 0.0
    run_threshold = float(metric_df["threshold"].median()) if not metric_df.empty else 0.5

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Test cases", meta["total_cases"])
    k2.metric("Case pass rate", f"{overall_pass_rate:.0%}")
    k3.metric("Avg metric score", f"{avg_score:.0%}")
    if meta.get("run_duration"):
        k4.metric("Last run duration", f"{meta['run_duration']:.0f}s")
    else:
        k4.metric("Evaluation cost", f"${meta.get('evaluation_cost', 0):.4f}")

    if not summary_df.empty:
        gauge_cols = st.columns(len(summary_df))
        for column, (_, row) in zip(gauge_cols, summary_df.iterrows()):
            with column:
                st.plotly_chart(
                    _score_gauge(
                        row["display_score"],
                        row["metric"],
                        METRIC_COLORS.get(row["metric"], "#818cf8"),
                        threshold=run_threshold,
                        inverted=bool(row["inverted"]),
                    ),
                    use_container_width=True,
                )

    left, right = st.columns([1.1, 0.9])
    with left:
        st.plotly_chart(
            _metric_bar_chart(summary_df, threshold=run_threshold),
            use_container_width=True,
        )
    with right:
        st.plotly_chart(_pass_fail_donut(meta), use_container_width=True)

    mid_left, mid_right = st.columns(2)
    with mid_left:
        st.plotly_chart(_radar_chart(summary_df), use_container_width=True)
    with mid_right:
        pass_df = summary_df[["metric", "passes", "fails"]].melt(
            id_vars="metric", var_name="outcome", value_name="count"
        )
        pass_df["outcome"] = pass_df["outcome"].map({"passes": "Pass", "fails": "Fail"})
        fig = px.bar(
            pass_df,
            x="metric",
            y="count",
            color="outcome",
            barmode="group",
            color_discrete_map={"Pass": "#10b981", "Fail": "#ef4444"},
            title="Pass / Fail by Metric",
        )
        fig.update_layout(height=380, xaxis_title="", yaxis_title="Count", **CHART_LAYOUT)
        st.plotly_chart(fig, use_container_width=True)

    st.plotly_chart(_heatmap(details_df), use_container_width=True)

    st.subheader("Detailed results")
    for case_id in sorted(metric_df["case_id"].unique()):
        case_metrics = metric_df[metric_df["case_id"] == case_id]
        question = case_metrics.iloc[0]["full_question"]
        case_passed = bool((case_metrics["passed"]).all())
        status = "Passed" if case_passed else "Needs attention"
        icon = "✅" if case_passed else "⚠️"

        with st.expander(f"{icon} Case {case_id}: {question[:90]}{'...' if len(question) > 90 else ''}"):
            st.markdown(f"**Question:** {question}")
            for _, row in case_metrics.iterrows():
                score_pct = f"{row['score']:.0%}"
                result = "Pass" if row["passed"] else "Fail"
                st.markdown(f"- **{row['metric']}** — {score_pct} ({result})")
                if row["reason"]:
                    st.caption(row["reason"])
