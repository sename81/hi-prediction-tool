import streamlit as st
import pandas as pd
import pickle
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# PDF
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    Image,
    PageBreak,
)
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER

# =========================
# SESSION STATE
# =========================
if "pdf_bytes" not in st.session_state:
    st.session_state.pdf_bytes = None

if "report_df" not in st.session_state:
    st.session_state.report_df = None

if "report_name" not in st.session_state:
    st.session_state.report_name = ""

# =========================
# LOAD MODELS
# =========================
with open("hi_model.pkl", "rb") as f:
    models = pickle.load(f)

with open("feature_names.pkl", "rb") as f:
    feature_names = pickle.load(f)

with open("function_models.pkl", "rb") as f:
    function_models = pickle.load(f)

with open("behavioral_models.pkl", "rb") as f:
    behavioral_models = pickle.load(f)

# =========================
# CONFIG
# =========================
trait_columns = [
    "Helpful",
    "Cause Motivated",
    "Self-Improvement",
    "Enthusiastic",
    "Open / reflective",
    "Wants Capable Leader",
    "Self-Motivated",
    "Takes Initiative",
    "Wants Recognition",
    "Wants Stable Career",
    "Wants Challenge",
    "Self-Acceptance",
    "Diplomatic",
    "Flexible",
    "Wants Frankness",
    "Tolerance Of Bluntness",
    "Planning",
    "Outgoing",
    "Analyzes Pitfalls",
    "Enlists Cooperation",
    "Wants High Pay",
    "Risking",
    "Wants Autonomy",
    "Organized",
    "Wants To Lead",
    "Optimistic",
    "Persistent",
    "Experimenting",
    "Assertive",
    "Analytical",
    "Manages Stress Well",
    "Systematic",
    "Comfort With Conflict",
    "Tempo",
    "Intuitive",
    "Authoritative",
    "Collaborative",
    "Tolerance Of Structure",
    "Influencing",
    "Frank",
    "Certain",
    "Enforcing",
    "Wants Diplomacy",
    "Relaxed",
    "Precise",
    "Warmth / empathy",
]

SECTION_ORDER = [
    "Traits",
    "Employment Expectations",
    "Task Preferences",
    "Interests",
    "Work Environment Preferences",
    "Behavioral Competencies",
    "Functions",
]

BEHAVIORAL_COLUMNS = [
    "Organizational Compatibility",
    "Handles Conflict",
    "Coaching",
    "People Oriented",
    "Innovative",
    "Interpersonal Skills",
    "Handles Autonomy",
    "Provides Direction",
    "Receives Correction",
    "Doesn't Need Structure",
    "Tolerance Of Evasiveness",
    "Negotiating",
    "Effective Enforcing",
    "Judgment (strategic)",
    "Self-Employed",
]

INTEREST_MAP = {
    "Writing / language": "Writing or language",
    "Children": "Children",
    "Animals": "Animals",
    "Food": "Making or serving food",
    "Sports": "Sports",
    "Travel": "Traveling",
    "Science": "Some types of science (any type)",
    "Health / medicine": "Health or medicine",
    "Finance / business": "Finance or business",
    "Electronics": "Electronics",
    "Plants": "Plants, trees or farming",
    "Entertainment": "Acting, dancing, singing or entertaining",
    "Selling": "Convincing people to buy something",
    "Psychology": "Trying to understand how and why people think and act the way they do",
    "Physical Science": "Physical science such as physics and chemistry",
    "Manufacturing": "Knowing how different things are made or manufactured",
    "Legal Matters": "The rules and laws that relate to society and business",
    "Biology": "Knowing how living things function, develop or reproduce",
    "Medical Science": "Medical science - diagnosis and treatment",
    "Computer Software": "Knowing how computer software works",
    "Computer Hardware": "Knowing how computer or computer related hardware works",
}

# =========================
# CHART DEFINITIONS
# Top, Right, Bottom, Left
# =========================
CHART_DEFINITIONS = [
    {
        "title": "OUTLOOK",
        "top_label": "Certain",
        "top_source": "Certain",
        "right_label": "Optimistic",
        "right_source": "Optimistic",
        "bottom_label": "Open / reflective",
        "bottom_source": "Open / reflective",
        "left_label": "Outgoing",
        "left_source": "Outgoing",
    },
    {
        "title": "DECISIONS",
        "top_label": "Problem Solving",
        "top_source": ["Analytical", "Analyzes Pitfalls"],
        "right_label": "Collaborative",
        "right_source": "Collaborative",
        "bottom_label": "Intuitive",
        "bottom_source": "Intuitive",
        "left_label": "Authoritative",
        "left_source": "Authoritative",
    },
    {
        "title": "INNOVATION",
        "top_label": "Persistent",
        "top_source": "Persistent",
        "right_label": "Tempo",
        "right_source": "Tempo",
        "bottom_label": "Experimenting",
        "bottom_source": "Experimenting",
        "left_label": "Risking",
        "left_source": "Risking",
    },
    {
        "title": "COMMUNICATION",
        "top_label": "Frank",
        "top_source": "Frank",
        "right_label": "Tolerance Of Bluntness",
        "right_source": "Tolerance Of Bluntness",
        "bottom_label": "Diplomatic",
        "bottom_source": "Diplomatic",
        "left_label": "Influencing",
        "left_source": "Influencing",
    },
    {
        "title": "POWER",
        "top_label": "Assertive",
        "top_source": "Assertive",
        "right_label": "Wants Capable Leader",
        "right_source": "Wants Capable Leader",
        "bottom_label": "Helpful",
        "bottom_source": "Helpful",
        "left_label": "Takes Authority",
        "left_source": ["Assertive", "Wants To Lead", "Authoritative"],
    },
    {
        "title": "MOTIVATION",
        "top_label": "Self-Motivated",
        "top_source": "Self-Motivated",
        "right_label": "Cause Motivated",
        "right_source": "Cause Motivated",
        "bottom_label": "Stress Management",
        "bottom_source": "Manages Stress Well",
        "left_label": "Wants High Pay",
        "left_source": "Wants High Pay",
    },
    {
        "title": "SUPPORT",
        "top_label": "Self-Acceptance",
        "top_source": "Self-Acceptance",
        "right_label": "Wants Recognition",
        "right_source": "Wants Recognition",
        "bottom_label": "Self-Improvement",
        "bottom_source": "Self-Improvement",
        "left_label": "Warmth / empathy",
        "left_source": "Warmth / empathy",
    },
    {
        "title": "ORGANIZATION",
        "top_label": "Organized",
        "top_source": "Organized",
        "right_label": "Tolerance Of Structure",
        "right_source": "Tolerance Of Structure",
        "bottom_label": "Flexible",
        "bottom_source": "Flexible",
        "left_label": "Precise",
        "left_source": "Precise",
    },
    {
        "title": "LEADERSHIP",
        "top_label": "Provides Direction",
        "top_source": "Provides Direction",
        "right_label": "Enforcing",
        "right_source": "Enforcing",
        "bottom_label": "Planning",
        "bottom_source": "Planning",
        "left_label": "Handles Conflict",
        "left_source": "Handles Conflict",
    },
]

# =========================
# HELPERS
# =========================
import math

def amplify_shape(score, k_high=1.6, k_low=2.5):
    centered = (score - 5) / 5

    if centered >= 0:
        boosted = math.tanh(centered * k_high)
    else:
        boosted = math.tanh(centered * k_low)

    return max(0, min(10, boosted * 5 + 5))

def selective_amplify(row):
    if row["Section"] in [
        "Traits",
        "Employment Expectations",
        "Task Preferences",
        "Interests",
        "Work Environment Preferences"
    ]:
        return row["Score"]  # no change

    elif row["Section"] in ["Behavioral Competencies", "Functions"]:
        return amplify_shape(row["Score"], k_high=1.6, k_low=2.5)

    else:
        return row["Score"]

def clamp_score(value: float) -> float:
    return max(0.0, min(10.0, float(value)))

def get_metric_score(df: pd.DataFrame, trait_name: str, default: float = 5.0) -> float:
    row = df[df["Trait Name"] == trait_name]
    if not row.empty:
        return float(row["Score"].iloc[0])
    return default

def redistribute_scores(values):
    import numpy as np

    ranks = np.argsort(np.argsort(values))  # get rank positions (0 to n-1)
    n = len(values)

    # map ranks into a wider range (3 to 9)
    scaled = 3 + (ranks / (n - 1)) * 6

    return scaled

def resolve_source_value(df: pd.DataFrame, source, default: float = 5.0) -> float:
    if isinstance(source, str):
        return get_metric_score(df, source, default=default)
    if isinstance(source, (list, tuple)):
        vals = [get_metric_score(df, s, default=default) for s in source]
        return sum(vals) / len(vals) if vals else default
    return default


def score_to_axis_length(score: float) -> float:
    # Keeps bars comfortably within chart box
    return (clamp_score(score) / 10.0) * 6.4


def draw_quadrant_chart(
    top_value,
    right_value,
    bottom_value,
    left_value,
    top_label,
    right_label,
    bottom_label,
    left_label,
    title,
    filename,
):
    fig, ax = plt.subplots(figsize=(4.2, 4.6))
    ax.set_xlim(-10.5, 10.5)
    ax.set_ylim(-12.0, 12.5)
    ax.axis("off")
    fig.patch.set_facecolor("white")

    # Title band
    ax.add_patch(Rectangle((-8.7, 9.5), 17.4, 1.8, facecolor="#2F67B2", edgecolor="none"))
    ax.text(0, 10.4, title, ha="center", va="center", color="white", fontsize=12, fontweight="bold")

    # Main box
    ax.add_patch(Rectangle((-8.0, -8.0), 16.0, 16.0, facecolor="white", edgecolor="black", linewidth=1.2))

    # Cross axes
    ax.plot([-8.0, 8.0], [0, 0], color="black", linewidth=1.0, zorder=1)
    ax.plot([0, 0], [-8.0, 8.0], color="black", linewidth=1.0, zorder=1)

    bar_color = "#8CD4E8"
    edge_color = "#2F67B2"

    lt = score_to_axis_length(left_value)
    rt = score_to_axis_length(right_value)
    tp = score_to_axis_length(top_value)
    bt = score_to_axis_length(bottom_value)

    # Bars
    ax.plot([-lt, 0], [0, 0], linewidth=10, solid_capstyle="butt", color=bar_color, zorder=3)
    ax.plot([0, rt], [0, 0], linewidth=10, solid_capstyle="butt", color=bar_color, zorder=3)
    ax.plot([0, 0], [0, tp], linewidth=10, solid_capstyle="butt", color=bar_color, zorder=3)
    ax.plot([0, 0], [-bt, 0], linewidth=10, solid_capstyle="butt", color=bar_color, zorder=3)

    ax.plot([-lt, 0], [0, 0], linewidth=2.2, solid_capstyle="butt", color=edge_color, zorder=4)
    ax.plot([0, rt], [0, 0], linewidth=2.2, solid_capstyle="butt", color=edge_color, zorder=4)
    ax.plot([0, 0], [0, tp], linewidth=2.2, solid_capstyle="butt", color=edge_color, zorder=4)
    ax.plot([0, 0], [-bt, 0], linewidth=2.2, solid_capstyle="butt", color=edge_color, zorder=4)

    # Value labels
    ax.text(-lt / 2 if lt > 0 else -0.2, 0.35, f"{round(left_value):.0f}", ha="center", va="center", fontsize=10, fontweight="bold")
    ax.text(rt / 2 if rt > 0 else 0.2, 0.35, f"{round(right_value):.0f}", ha="center", va="center", fontsize=10, fontweight="bold")
    ax.text(0.35, tp / 2 if tp > 0 else 0.2, f"{round(top_value):.0f}", ha="center", va="center", fontsize=10, fontweight="bold")
    ax.text(0.35, -bt / 2 if bt > 0 else -0.2, f"{round(bottom_value):.0f}", ha="center", va="center", fontsize=10, fontweight="bold")

    # Axis labels
    ax.text(0, 8.7, top_label, ha="center", va="center", fontsize=10)
    ax.text(8.9, 0, right_label, ha="center", va="center", fontsize=10, rotation=270)
    ax.text(0, -9.0, bottom_label, ha="center", va="center", fontsize=10)
    ax.text(-8.9, 0, left_label, ha="center", va="center", fontsize=10, rotation=90)

    plt.tight_layout()
    plt.savefig(filename, dpi=180, bbox_inches="tight")
    plt.close()


def generate_all_charts(df: pd.DataFrame):
    chart_files = []

    for i, chart in enumerate(CHART_DEFINITIONS):
        filename = f"chart_{i}.png"

        top_value = resolve_source_value(df, chart["top_source"])
        right_value = resolve_source_value(df, chart["right_source"])
        bottom_value = resolve_source_value(df, chart["bottom_source"])
        left_value = resolve_source_value(df, chart["left_source"])

        draw_quadrant_chart(
            top_value=top_value,
            right_value=right_value,
            bottom_value=bottom_value,
            left_value=left_value,
            top_label=chart["top_label"],
            right_label=chart["right_label"],
            bottom_label=chart["bottom_label"],
            left_label=chart["left_label"],
            title=chart["title"],
            filename=filename,
        )

        chart_files.append(filename)

    return chart_files


def add_page_number(canvas, doc):
    page_num = canvas.getPageNumber()
    canvas.setFont("Helvetica", 9)
    canvas.setFillColor(colors.grey)
    canvas.drawCentredString(doc.pagesize[0] / 2.0, 18, f"Page {page_num}")


def build_report_dataframe(df_input: pd.DataFrame) -> pd.DataFrame:
    X_input = pd.DataFrame([df_input["score"].tolist()], columns=feature_names)
    employment_values = df_input["score"].iloc[-10:].tolist()

    results = {}
    for col, model in models.items():
        results[col] = float(model.predict(X_input)[0])

    df = pd.DataFrame(list(results.items()), columns=["Trait", "Score"])

    employment_names = [
        "Wants Advancement",
        "Wants Opinions Valued",
        "Wants Development",
        "Wants Social Opportunities",
        "Wants Work/Life Balance",
        "Wants Appreciation",
        "Wants To Be Informed",
        "Wants Personal Help",
        "Wants Flexible Work Time",
        "Wants Quick Pay Increases",
    ]

    emp_rows = []
    for i, name in enumerate(employment_names):
        emp_rows.append({
            "Trait": f"Employment Expectations | {name}",
            "Score": float(employment_values[i]),
        })

    df = pd.concat([df, pd.DataFrame(emp_rows)], ignore_index=True)

    df["Section"] = df["Trait"].apply(lambda x: x.split(" | ")[0])
    df["Trait Name"] = df["Trait"].apply(lambda x: x.split(" | ")[1])


    # Override Interests with direct Type 1 mapping
    for output_label, input_label in INTEREST_MAP.items():
        mask = (df["Section"] == "Interests") & (df["Trait Name"] == output_label)

        if mask.any():
            value = df_input.loc[
                df_input["statement"].str.strip().str.lower() == input_label.strip().lower(),
                "score"
            ].values

            if len(value) > 0:
                df.loc[mask, "Score"] = float(value[0])

    trait_scores = dict(zip(df["Trait Name"], df["Score"]))
    trait_vector = [trait_scores.get(t, 5.0) for t in trait_columns]

    # Behavioral boosting
    for idx, row in df.iterrows():
        if row["Section"] == "Behavioral Competencies":
            name = row["Trait Name"]
            if name in behavioral_models:
                base_score = df.at[idx, "Score"]

                raw_behavior = behavioral_models[name].predict([trait_vector])[0]
                behavior_score = 5 + (raw_behavior - 5) * 1.5

                df.at[idx, "Score"] = clamp_score(0.5 * base_score + 0.5 * behavior_score)

    # Functions
    function_rows = []
    for fname, model in function_models.items():
        fscore = clamp_score(model.predict([trait_vector])[0])
        function_rows.append({
            "Trait": f"Functions | {fname}",
            "Score": round(fscore, 1),
        })

    df = df[df["Section"] != "Functions"]
    df = pd.concat([df, pd.DataFrame(function_rows)], ignore_index=True)

    df["Section"] = df["Trait"].apply(lambda x: x.split(" | ")[0])
    df["Trait Name"] = df["Trait"].apply(lambda x: x.split(" | ")[1])

    return df


def generate_pdf(df: pd.DataFrame, candidate_name: str) -> str:
    chart_files = generate_all_charts(df)

    doc = SimpleDocTemplate(
        "report.pdf",
        rightMargin=36,
        leftMargin=36,
        topMargin=36,
        bottomMargin=36,
    )

    styles = getSampleStyleSheet()

    report_title_style = ParagraphStyle(
        "ReportTitle",
        parent=styles["Title"],
        fontName="Helvetica-Bold",
        fontSize=20,
        leading=24,
        alignment=TA_CENTER,
        textColor=colors.black,
        spaceAfter=6,
    )

    name_style = ParagraphStyle(
        "CandidateName",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=12,
        leading=14,
        alignment=TA_CENTER,
        textColor=colors.black,
        spaceAfter=2,
    )

    date_style = ParagraphStyle(
        "ReportDate",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=10,
        leading=12,
        alignment=TA_CENTER,
        textColor=colors.grey,
        spaceAfter=14,
    )

    section_header_style = ParagraphStyle(
        "SectionHeader",
        parent=styles["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=14,
        leading=18,
        textColor=colors.black,
        spaceAfter=6,
    )

    content = []
    display_name = candidate_name.strip() if candidate_name.strip() else "Candidate"

    content.append(Paragraph("RE HI Report", report_title_style))
    content.append(Paragraph(display_name, name_style))
    content.append(Paragraph(f"Report Date: {datetime.today().strftime('%d %b %Y')}", date_style))

    for section in SECTION_ORDER:
        sub_df = df[df["Section"] == section].copy().sort_values("Score", ascending=False)

        if section == "Behavioral Competencies":
            content.append(PageBreak())

        content.append(Paragraph(section, section_header_style))
        content.append(Spacer(1, 6))

        table_data = [["Trait", "Score"]]

        style_cmds = [
            ("BACKGROUND", (0, 0), (-1, 0), colors.grey),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 10),
            ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
            ("ALIGN", (0, 0), (0, -1), "LEFT"),
            ("ALIGN", (1, 0), (1, -1), "CENTER"),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("GRID", (0, 0), (-1, -1), 0.25, colors.black),
            ("BOTTOMPADDING", (0, 0), (-1, 0), 6),
            ("TOPPADDING", (0, 0), (-1, 0), 6),
        ]

        trait_count = len(sub_df)

        for i, (_, row) in enumerate(sub_df.iterrows()):
            table_data.append([row["Trait Name"], f"{row['Score']:.1f}"])

            if section == "Traits":
                if i < 5:
                    style_cmds.append(("BACKGROUND", (0, i + 1), (-1, i + 1), colors.lightgrey))
                    style_cmds.append(("FONTNAME", (0, i + 1), (-1, i + 1), "Helvetica-Bold"))
                if i >= max(0, trait_count - 5):
                    style_cmds.append(("BACKGROUND", (0, i + 1), (-1, i + 1), colors.HexColor("#FDEDEC")))

        table = Table(table_data, colWidths=[300, 80])
        table.setStyle(TableStyle(style_cmds))

        content.append(table)
        content.append(Spacer(1, 12))
    
    # =========================
    # 9 charts in ONE page
    # =========================
    content.append(PageBreak())

    content.append(Paragraph("Main Graph and Narrative", report_title_style))
    content.append(Spacer(1, 8))

    rows = []
    for i in range(0, len(chart_files), 3):
        row = [Image(chart_files[i + j], width=150, height=165) for j in range(3)]
        rows.append(row)

    chart_table = Table(rows, colWidths=[170, 170, 170])
    chart_table.setStyle(TableStyle([
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ]))

    content.append(chart_table)

    doc.build(content, onFirstPage=add_page_number, onLaterPages=add_page_number)
    return "report.pdf"


def style_traits_dataframe(sub: pd.DataFrame):
    top_idx = sub.head(5).index
    bottom_idx = sub.tail(5).index

    def row_style(row):
        if row.name in top_idx:
            return ["background-color:#e6e6e6; font-weight:bold;", "background-color:#e6e6e6; font-weight:bold; text-align:center;"]
        if row.name in bottom_idx:
            return ["background-color:#fdecea;", "background-color:#fdecea; text-align:center;"]
        return ["", "text-align:center;"]

    styler = sub[["Trait Name", "Score"]].style.apply(row_style, axis=1)
    styler = styler.set_properties(subset=["Score"], **{"text-align": "center"})
    return styler


def style_regular_dataframe(sub: pd.DataFrame):
    styler = sub[["Trait Name", "Score"]].style.set_properties(subset=["Score"], **{"text-align": "center"})
    return styler


# =========================
# UI
# =========================
st.title("HI Prediction Tool")

candidate_name = st.text_input("Enter Candidate Name")
uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

if uploaded_file is not None and st.button("Generate"):
    df_input = pd.read_csv(uploaded_file)[["statement", "score"]].copy().reset_index(drop=True)

    report_df = build_report_dataframe(df_input)

    report_df["Score"] = report_df.apply(selective_amplify, axis=1)

    # reshape behavioral distribution
    mask = report_df["Section"] == "Behavioral Competencies"

    if mask.any():
        values = report_df.loc[mask, "Score"].values
        reshaped = redistribute_scores(values)
        report_df.loc[mask, "Score"] = reshaped
    
    report_name = candidate_name.strip() if candidate_name.strip() else "Candidate"

    pdf_file = generate_pdf(report_df, report_name)
    with open(pdf_file, "rb") as f:
        st.session_state.pdf_bytes = f.read()

    st.session_state.report_df = report_df
    st.session_state.report_name = report_name

# =========================
# DISPLAY LAST RESULT
# =========================
if st.session_state.report_df is not None:
    df = st.session_state.report_df.copy()
    display_name = st.session_state.report_name

    st.subheader(f"Results for {display_name}")

    for section in SECTION_ORDER:
        sub = df[df["Section"] == section].copy().sort_values("Score", ascending=False)
        sub["Score"] = sub["Score"].map(lambda x: f"{x:.1f}")

        st.markdown(f"### {section}")

        if section == "Traits":
            st.dataframe(style_traits_dataframe(sub), use_container_width=True)
        else:
            st.dataframe(style_regular_dataframe(sub), use_container_width=True)

    st.markdown("### Main Graph and Narrative")
    chart_files = generate_all_charts(df)

    for row_start in range(0, len(chart_files), 3):
        cols = st.columns(3)
        for i, chart_file in enumerate(chart_files[row_start:row_start + 3]):
            with cols[i]:
                st.image(chart_file, use_container_width=True)

# =========================
# DOWNLOAD BUTTON (BOTTOM)
# =========================
if st.session_state.pdf_bytes is not None:
    filename = f"{st.session_state.report_name}_HI_Report.pdf" if st.session_state.report_name else "HI_Report.pdf"

    st.download_button(
        "Download PDF",
        data=st.session_state.pdf_bytes,
        file_name=filename,
        mime="application/pdf",
    )
