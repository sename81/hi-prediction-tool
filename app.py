import streamlit as st
import pandas as pd
import pickle
from datetime import datetime
import matplotlib.pyplot as plt

# PDF
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    Image,
    PageBreak,
    KeepTogether,
)
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER

# =========================
# LOAD MODEL + FEATURES
# =========================
with open("hi_model.pkl", "rb") as f:
    models = pickle.load(f)

with open("feature_names.pkl", "rb") as f:
    feature_names = pickle.load(f)

# =========================
# SECTION ORDER
# =========================
SECTION_ORDER = [
    "Traits",
    "Employment Expectations",
    "Task Preferences",
    "Interests",
    "Work Environment Preferences",
    "Behavioral Competencies",
    "Functions"
]

# =========================
# FUNCTION MAP
# =========================
FUNCTION_MAP = {
    "Administration - General": ["organ", "plan", "system", "precise", "clerical"],
    "Sales - Cold Calling": ["influenc", "assert", "challenge", "public speaking", "negotiat"],
    "Management - Upper": ["lead", "direct", "strategic", "authoritative", "decision"],
    "Supervisory": ["lead", "direct", "coach", "enforc", "conflict"],
    "Management - Middle": ["lead", "organ", "plan", "coach", "responsib"],
    "Customer Service - Friendly": ["help", "warmth", "people", "interpersonal", "friendly"],
    "Technical": ["analytic", "system", "precise", "technical", "mechanical", "comput"]
}

# =========================
# CHART HELPERS
# =========================
def get_trait_score(df, name, default=5.0):
    row = df[df["Trait Name"] == name]
    return float(row["Score"].values[0]) if not row.empty else default


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
    filename
):
    fig, ax = plt.subplots(figsize=(5.2, 5.2))

    ax.set_xlim(-10, 10)
    ax.set_ylim(-10, 10)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_facecolor("white")
    fig.patch.set_facecolor("white")

    for spine in ax.spines.values():
        spine.set_linewidth(1.6)
        spine.set_color("black")

    ax.axhline(0, color="black", linewidth=1.0, zorder=1)
    ax.axvline(0, color="black", linewidth=1.0, zorder=1)

    bar_color = "#67B7E3"

    ax.plot([-left_value, 0], [0, 0], linewidth=10, solid_capstyle="butt", color=bar_color, zorder=3)
    ax.plot([0, right_value], [0, 0], linewidth=10, solid_capstyle="butt", color=bar_color, zorder=3)
    ax.plot([0, 0], [0, top_value], linewidth=10, solid_capstyle="butt", color=bar_color, zorder=3)
    ax.plot([0, 0], [-bottom_value, 0], linewidth=10, solid_capstyle="butt", color=bar_color, zorder=3)

    # Value labels
    if left_value > 0:
        ax.text(-left_value / 2, 0.45, f"{left_value:.1f}", ha="center", va="center",
                fontsize=10, color="black", fontweight="bold", zorder=4)
    if right_value > 0:
        ax.text(right_value / 2, 0.45, f"{right_value:.1f}", ha="center", va="center",
                fontsize=10, color="black", fontweight="bold", zorder=4)
    if top_value > 0:
        ax.text(0.45, top_value / 2, f"{top_value:.1f}", ha="center", va="center",
                fontsize=10, color="black", fontweight="bold", zorder=4)
    if bottom_value > 0:
        ax.text(0.45, -bottom_value / 2, f"{bottom_value:.1f}", ha="center", va="center",
                fontsize=10, color="black", fontweight="bold", zorder=4)

    ax.text(0, 10.9, top_label, ha="center", va="center", fontsize=11)
    ax.text(10.9, 0, right_label, ha="center", va="center", fontsize=11, rotation=270)
    ax.text(0, -11.2, bottom_label, ha="center", va="center", fontsize=11)
    ax.text(-10.9, 0, left_label, ha="center", va="center", fontsize=11, rotation=90)

    plt.tight_layout(rect=[0.04, 0.03, 0.96, 0.95])
    plt.savefig(filename, dpi=180, bbox_inches="tight")
    plt.close()


def generate_innovation_chart(df, filename="innovation.png"):
    planning = get_trait_score(df, "Planning")
    persistent = get_trait_score(df, "Persistent")
    tempo = get_trait_score(df, "Tempo")
    experimenting = get_trait_score(df, "Experimenting")

    draw_quadrant_chart(
        top_value=persistent,
        right_value=tempo,
        bottom_value=experimenting,
        left_value=planning,
        top_label="Persistent",
        right_label="Tempo",
        bottom_label="Experimenting",
        left_label="Planning",
        title="INNOVATION",
        filename=filename
    )


def generate_support_chart(df, filename="support.png"):
    wants_autonomy = get_trait_score(df, "Wants Autonomy")
    self_acceptance = get_trait_score(df, "Self-Acceptance")
    wants_recognition = get_trait_score(df, "Wants Recognition")
    self_improvement = get_trait_score(df, "Self-Improvement")

    draw_quadrant_chart(
        top_value=self_acceptance,
        right_value=wants_recognition,
        bottom_value=self_improvement,
        left_value=wants_autonomy,
        top_label="Self-Acceptance",
        right_label="Wants Recognition",
        bottom_label="Self-Improvement",
        left_label="Wants Autonomy",
        title="SUPPORT",
        filename=filename
    )

# =========================
# PAGE NUMBER HELPER
# =========================
def add_page_number(canvas, doc):
    page_num = canvas.getPageNumber()
    canvas.setFont("Helvetica", 9)
    canvas.setFillColor(colors.grey)
    canvas.drawCentredString(doc.pagesize[0] / 2.0, 18, f"Page {page_num}")

# =========================
# PDF GENERATION
# =========================
def generate_pdf(df, candidate_name):
    generate_innovation_chart(df)
    generate_support_chart(df)

    doc = SimpleDocTemplate(
        "report.pdf",
        rightMargin=36,
        leftMargin=36,
        topMargin=36,
        bottomMargin=36
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
        spaceAfter=6
    )

    name_style = ParagraphStyle(
        "CandidateName",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=12,
        leading=14,
        alignment=TA_CENTER,
        textColor=colors.black,
        spaceAfter=2
    )

    date_style = ParagraphStyle(
        "ReportDate",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=10,
        leading=12,
        alignment=TA_CENTER,
        textColor=colors.grey,
        spaceAfter=14
    )

    section_header_style = ParagraphStyle(
        "SectionHeader",
        parent=styles["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=14,
        leading=18,
        textColor=colors.black,
        spaceAfter=6
    )

    chart_header_style = ParagraphStyle(
        "ChartHeader",
        parent=styles["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=13,
        leading=16,
        alignment=TA_CENTER,
        textColor=colors.black,
        spaceAfter=6
    )

    content = []

    display_name = candidate_name.strip() if candidate_name.strip() else "Candidate"

    content.append(Paragraph("RE HI Report", report_title_style))
    content.append(Paragraph(display_name, name_style))
    content.append(Paragraph(f"Report Date: {datetime.today().strftime('%d %b %Y')}", date_style))

    for section in SECTION_ORDER:
        sub_df = df[df["Section"] == section].copy().sort_values("Score", ascending=False)

        # Force Behavioral Competencies to a new page
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
            ("ALIGN", (1, 0), (1, -1), "CENTER"),  # middle justify score column
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("GRID", (0, 0), (-1, -1), 0.25, colors.black),
            ("BOTTOMPADDING", (0, 0), (-1, 0), 6),
            ("TOPPADDING", (0, 0), (-1, 0), 6),
        ]

        for i, (_, row) in enumerate(sub_df.iterrows()):
            table_data.append([row["Trait Name"], f"{row['Score']:.1f}"])

            if section == "Traits" and i < 5:
                style_cmds.append(("BACKGROUND", (0, i + 1), (-1, i + 1), colors.lightgrey))
                style_cmds.append(("FONTNAME", (0, i + 1), (-1, i + 1), "Helvetica-Bold"))

        table = Table(table_data, colWidths=[300, 80])
        table.setStyle(TableStyle(style_cmds))

        content.append(table)
        content.append(Spacer(1, 12))

    # Final charts page: both charts on one page, no duplication
    content.append(PageBreak())
    content.append(Paragraph("Profile Charts", section_header_style))
    content.append(Spacer(1, 8))

    chart_block = [
        Paragraph("Innovation", chart_header_style),
        Image("innovation.png", width=260, height=260),
        Spacer(1, 10),
        Paragraph("Support", chart_header_style),
        Image("support.png", width=260, height=260),
    ]
    content.append(KeepTogether(chart_block))

    doc.build(content, onFirstPage=add_page_number, onLaterPages=add_page_number)
    return "report.pdf"

# =========================
# UI
# =========================
st.title("HI Prediction Tool")

candidate_name = st.text_input("Enter Candidate Name")
uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

if uploaded_file:
    df_input = pd.read_csv(uploaded_file)
    df_input = df_input[["statement", "score"]].copy().reset_index(drop=True)

    X_input = pd.DataFrame([df_input["score"].tolist()], columns=feature_names)
    employment_values = df_input["score"].iloc[-10:].tolist()

    if st.button("Predict"):
        results = {}
        for col, model in models.items():
            results[col] = float(model.predict(X_input)[0])

        df = pd.DataFrame(list(results.items()), columns=["Trait", "Score"])

        names = [
            "Wants Advancement",
            "Wants Opinions Valued",
            "Wants Development",
            "Wants Social Opportunities",
            "Wants Work/Life Balance",
            "Wants Appreciation",
            "Wants To Be Informed",
            "Wants Personal Help",
            "Wants Flexible Work Time",
            "Wants Quick Pay Increases"
        ]

        emp_rows = []
        for i, n in enumerate(names):
            emp_rows.append({
                "Trait": f"Employment Expectations | {n}",
                "Score": float(employment_values[i])
            })

        df = pd.concat([df, pd.DataFrame(emp_rows)], ignore_index=True)

        df["Section"] = df["Trait"].apply(lambda x: x.split(" | ")[0])
        df["Trait Name"] = df["Trait"].apply(lambda x: x.split(" | ")[1])

        trait_scores = dict(zip(df["Trait Name"], df["Score"]))
        function_rows = []

        for fname, keywords in FUNCTION_MAP.items():
            matched_scores = []

            for trait, score in trait_scores.items():
                for kw in keywords:
                    if kw in trait.lower():
                        matched_scores.append(score)

            fscore = sum(matched_scores) / len(matched_scores) if matched_scores else 5.0

            function_rows.append({
                "Trait": f"Functions | {fname}",
                "Score": round(fscore, 1)
            })

        df = df[df["Section"] != "Functions"]
        df_functions = pd.DataFrame(function_rows)
        df = pd.concat([df, df_functions], ignore_index=True)

        df["Section"] = df["Trait"].apply(lambda x: x.split(" | ")[0])
        df["Trait Name"] = df["Trait"].apply(lambda x: x.split(" | ")[1])

        generate_innovation_chart(df)
        generate_support_chart(df)

        st.subheader(f"Results for {candidate_name}")

        for section in SECTION_ORDER:
            sub = df[df["Section"] == section].copy().sort_values("Score", ascending=False)
            sub["Score"] = sub["Score"].map(lambda x: f"{x:.1f}")

            st.markdown(f"### {section}")

            if section == "Traits":
                top5_idx = sub.head(5).index

                def highlight(row):
                    if row.name in top5_idx:
                        return ["font-weight: bold; background-color: #f0f0f0"] * len(row)
                    return [""] * len(row)

                st.dataframe(
                    sub[["Trait Name", "Score"]].style.apply(highlight, axis=1),
                    use_container_width=True
                )
            else:
                st.dataframe(sub[["Trait Name", "Score"]], use_container_width=True)

        st.markdown("### Profile Charts")
        col1, col2 = st.columns(2)
        with col1:
            st.image("innovation.png", caption="Innovation", use_container_width=True)
        with col2:
            st.image("support.png", caption="Support", use_container_width=True)

        pdf_file = generate_pdf(df, candidate_name)

        with open(pdf_file, "rb") as f:
            st.download_button(
                "Download PDF",
                f,
                file_name=f"{candidate_name}_HI_Report.pdf",
                mime="application/pdf"
            )