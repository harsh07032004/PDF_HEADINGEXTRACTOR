"""
ml/generate_pdfs.py

Programmatically generate synthetic training PDFs with perfect ground-truth
labels using PyMuPDF (fitz).

Each PDF has a known, deterministic heading structure so the ML classifier
can learn the relationship between typographic features and heading levels.

Usage:
    python -m ml.generate_pdfs                    # generates 15 PDFs
    python -m ml.generate_pdfs --count 20         # custom count
    python -m ml.generate_pdfs --out ml/training_data
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import fitz  # PyMuPDF


# ── Document templates ─────────────────────────────────────────────────────────

# Each template defines a realistic document structure with H1, H2, H3 headings.
# Body text is generated between headings.

TEMPLATES = [
    {
        "title": "Machine Learning in Healthcare",
        "sections": [
            ("H1", "1. Introduction"),
            ("H2", "1.1 Background"),
            ("H2", "1.2 Motivation"),
            ("H1", "2. Related Work"),
            ("H2", "2.1 Deep Learning Approaches"),
            ("H3", "2.1.1 Convolutional Networks"),
            ("H3", "2.1.2 Recurrent Networks"),
            ("H2", "2.2 Traditional Methods"),
            ("H1", "3. Methodology"),
            ("H2", "3.1 Data Collection"),
            ("H2", "3.2 Feature Engineering"),
            ("H3", "3.2.1 Clinical Features"),
            ("H3", "3.2.2 Imaging Features"),
            ("H2", "3.3 Model Architecture"),
            ("H1", "4. Results"),
            ("H2", "4.1 Classification Accuracy"),
            ("H2", "4.2 Comparison with Baselines"),
            ("H1", "5. Discussion"),
            ("H1", "6. Conclusion"),
            ("H1", "References"),
        ],
    },
    {
        "title": "Annual Financial Report 2024",
        "sections": [
            ("H1", "Executive Summary"),
            ("H1", "Financial Highlights"),
            ("H2", "Revenue Overview"),
            ("H2", "Operating Expenses"),
            ("H3", "Personnel Costs"),
            ("H3", "Infrastructure"),
            ("H2", "Net Income"),
            ("H1", "Business Segments"),
            ("H2", "Consumer Products"),
            ("H2", "Enterprise Solutions"),
            ("H3", "Cloud Services"),
            ("H3", "On-Premise Software"),
            ("H2", "Professional Services"),
            ("H1", "Risk Factors"),
            ("H2", "Market Risk"),
            ("H2", "Regulatory Risk"),
            ("H1", "Outlook for 2025"),
            ("H1", "Appendix"),
        ],
    },
    {
        "title": "Software Engineering Best Practices",
        "sections": [
            ("H1", "INTRODUCTION"),
            ("H1", "CODE QUALITY"),
            ("H2", "Static Analysis"),
            ("H2", "Code Reviews"),
            ("H3", "Review Checklist"),
            ("H3", "Common Pitfalls"),
            ("H1", "TESTING STRATEGIES"),
            ("H2", "Unit Testing"),
            ("H2", "Integration Testing"),
            ("H2", "End-to-End Testing"),
            ("H3", "Test Automation Frameworks"),
            ("H1", "DEPLOYMENT"),
            ("H2", "Continuous Integration"),
            ("H2", "Continuous Deployment"),
            ("H3", "Blue-Green Deployments"),
            ("H3", "Canary Releases"),
            ("H1", "MONITORING"),
            ("H2", "Logging"),
            ("H2", "Alerting"),
            ("H1", "CONCLUSION"),
        ],
    },
    {
        "title": "Environmental Impact Assessment",
        "sections": [
            ("H1", "1 Project Description"),
            ("H2", "1.1 Site Location"),
            ("H2", "1.2 Project Scope"),
            ("H1", "2 Environmental Baseline"),
            ("H2", "2.1 Air Quality"),
            ("H2", "2.2 Water Resources"),
            ("H3", "2.2.1 Surface Water"),
            ("H3", "2.2.2 Groundwater"),
            ("H2", "2.3 Ecology"),
            ("H3", "2.3.1 Flora"),
            ("H3", "2.3.2 Fauna"),
            ("H1", "3 Impact Assessment"),
            ("H2", "3.1 Construction Phase"),
            ("H2", "3.2 Operational Phase"),
            ("H1", "4 Mitigation Measures"),
            ("H2", "4.1 Air Quality Measures"),
            ("H2", "4.2 Water Protection"),
            ("H1", "5 Monitoring Plan"),
            ("H1", "6 Conclusions"),
        ],
    },
    {
        "title": "User Interface Design Guidelines",
        "sections": [
            ("H1", "Design Principles"),
            ("H2", "Consistency"),
            ("H2", "Accessibility"),
            ("H3", "Color Contrast"),
            ("H3", "Screen Readers"),
            ("H2", "Responsiveness"),
            ("H1", "Typography"),
            ("H2", "Font Selection"),
            ("H2", "Hierarchy"),
            ("H3", "Heading Sizes"),
            ("H3", "Body Text"),
            ("H1", "Color System"),
            ("H2", "Primary Palette"),
            ("H2", "Semantic Colors"),
            ("H1", "Component Library"),
            ("H2", "Buttons"),
            ("H2", "Forms"),
            ("H3", "Text Inputs"),
            ("H3", "Dropdowns"),
            ("H2", "Cards"),
            ("H1", "Layout Patterns"),
        ],
    },
    {
        "title": "Clinical Trial Protocol",
        "sections": [
            ("H1", "Study Overview"),
            ("H2", "Study Objectives"),
            ("H2", "Study Design"),
            ("H1", "Eligibility Criteria"),
            ("H2", "Inclusion Criteria"),
            ("H2", "Exclusion Criteria"),
            ("H1", "Treatment Plan"),
            ("H2", "Dosing Schedule"),
            ("H3", "Dose Escalation"),
            ("H3", "Dose Modification"),
            ("H2", "Concomitant Medications"),
            ("H1", "Safety Monitoring"),
            ("H2", "Adverse Events"),
            ("H2", "Serious Adverse Events"),
            ("H3", "Reporting Requirements"),
            ("H1", "Statistical Analysis"),
            ("H2", "Sample Size"),
            ("H2", "Primary Endpoint"),
            ("H2", "Secondary Endpoints"),
            ("H1", "Ethical Considerations"),
        ],
    },
    {
        "title": "Data Privacy and Security Policy",
        "sections": [
            ("H1", "1. Purpose and Scope"),
            ("H1", "2. Definitions"),
            ("H1", "3. Data Classification"),
            ("H2", "3.1 Public Data"),
            ("H2", "3.2 Internal Data"),
            ("H2", "3.3 Confidential Data"),
            ("H3", "3.3.1 PII"),
            ("H3", "3.3.2 Financial Data"),
            ("H1", "4. Access Controls"),
            ("H2", "4.1 Authentication"),
            ("H2", "4.2 Authorization"),
            ("H3", "4.2.1 Role-Based Access"),
            ("H1", "5. Encryption"),
            ("H2", "5.1 Data at Rest"),
            ("H2", "5.2 Data in Transit"),
            ("H1", "6. Incident Response"),
            ("H2", "6.1 Detection"),
            ("H2", "6.2 Containment"),
            ("H2", "6.3 Recovery"),
            ("H1", "7. Compliance"),
        ],
    },
    {
        "title": "Product Requirements Document",
        "sections": [
            ("H1", "Executive Summary"),
            ("H1", "Problem Statement"),
            ("H2", "User Pain Points"),
            ("H2", "Market Gap"),
            ("H1", "Product Vision"),
            ("H1", "Requirements"),
            ("H2", "Functional Requirements"),
            ("H3", "User Registration"),
            ("H3", "Content Management"),
            ("H3", "Search and Discovery"),
            ("H2", "Non-Functional Requirements"),
            ("H3", "Performance"),
            ("H3", "Scalability"),
            ("H3", "Security"),
            ("H1", "User Stories"),
            ("H2", "Admin User Stories"),
            ("H2", "End User Stories"),
            ("H1", "Timeline"),
            ("H1", "Success Metrics"),
        ],
    },
    {
        "title": "Renewable Energy Systems Report",
        "sections": [
            ("H1", "Introduction"),
            ("H2", "Global Energy Landscape"),
            ("H2", "Renewable Energy Targets"),
            ("H1", "Solar Energy"),
            ("H2", "Photovoltaic Technology"),
            ("H3", "Monocrystalline Panels"),
            ("H3", "Polycrystalline Panels"),
            ("H2", "Solar Thermal"),
            ("H1", "Wind Energy"),
            ("H2", "Onshore Wind"),
            ("H2", "Offshore Wind"),
            ("H3", "Floating Turbines"),
            ("H1", "Energy Storage"),
            ("H2", "Battery Technologies"),
            ("H3", "Lithium-Ion"),
            ("H3", "Solid State"),
            ("H2", "Hydrogen Storage"),
            ("H1", "Grid Integration"),
            ("H1", "Policy Recommendations"),
        ],
    },
    {
        "title": "Operating System Architecture",
        "sections": [
            ("H1", "Chapter 1: Overview"),
            ("H2", "1.1 System Components"),
            ("H2", "1.2 Design Goals"),
            ("H1", "Chapter 2: Process Management"),
            ("H2", "2.1 Process Scheduling"),
            ("H3", "2.1.1 Round Robin"),
            ("H3", "2.1.2 Priority Scheduling"),
            ("H2", "2.2 Inter-Process Communication"),
            ("H1", "Chapter 3: Memory Management"),
            ("H2", "3.1 Virtual Memory"),
            ("H2", "3.2 Paging"),
            ("H3", "3.2.1 Page Tables"),
            ("H3", "3.2.2 TLB"),
            ("H1", "Chapter 4: File Systems"),
            ("H2", "4.1 Inode-based Systems"),
            ("H2", "4.2 Journaling"),
            ("H1", "Chapter 5: I/O Systems"),
            ("H1", "Chapter 6: Security"),
        ],
    },
    {
        "title": "Marketing Strategy 2025",
        "sections": [
            ("H1", "Market Analysis"),
            ("H2", "Target Demographics"),
            ("H2", "Competitive Landscape"),
            ("H3", "Direct Competitors"),
            ("H3", "Indirect Competitors"),
            ("H1", "Brand Positioning"),
            ("H2", "Value Proposition"),
            ("H2", "Brand Voice"),
            ("H1", "Channel Strategy"),
            ("H2", "Digital Channels"),
            ("H3", "Social Media"),
            ("H3", "Email Marketing"),
            ("H3", "Content Marketing"),
            ("H2", "Traditional Channels"),
            ("H1", "Budget Allocation"),
            ("H2", "Q1 Spend"),
            ("H2", "Q2 Spend"),
            ("H1", "KPIs and Metrics"),
            ("H1", "Risk Mitigation"),
        ],
    },
    {
        "title": "Database Design and Optimization",
        "sections": [
            ("H1", "Introduction"),
            ("H1", "Schema Design"),
            ("H2", "Normalization"),
            ("H3", "First Normal Form"),
            ("H3", "Second Normal Form"),
            ("H3", "Third Normal Form"),
            ("H2", "Denormalization Strategies"),
            ("H1", "Indexing"),
            ("H2", "B-Tree Indexes"),
            ("H2", "Hash Indexes"),
            ("H2", "Composite Indexes"),
            ("H1", "Query Optimization"),
            ("H2", "Query Plans"),
            ("H2", "Join Strategies"),
            ("H3", "Nested Loop"),
            ("H3", "Hash Join"),
            ("H1", "Replication"),
            ("H2", "Master-Slave"),
            ("H2", "Multi-Master"),
            ("H1", "Backup and Recovery"),
        ],
    },
    {
        "title": "Nutrition and Wellness Guide",
        "sections": [
            ("H1", "Macronutrients"),
            ("H2", "Proteins"),
            ("H3", "Animal Proteins"),
            ("H3", "Plant Proteins"),
            ("H2", "Carbohydrates"),
            ("H3", "Simple Carbs"),
            ("H3", "Complex Carbs"),
            ("H2", "Fats"),
            ("H1", "Micronutrients"),
            ("H2", "Vitamins"),
            ("H2", "Minerals"),
            ("H1", "Meal Planning"),
            ("H2", "Daily Caloric Intake"),
            ("H2", "Sample Meal Plans"),
            ("H3", "Breakfast Options"),
            ("H3", "Lunch Options"),
            ("H3", "Dinner Options"),
            ("H1", "Exercise Integration"),
            ("H1", "Supplements"),
        ],
    },
    {
        "title": "Research Methodology Handbook",
        "sections": [
            ("H1", "PART I: FOUNDATIONS"),
            ("H2", "Research Paradigms"),
            ("H2", "Ethics in Research"),
            ("H1", "PART II: QUANTITATIVE METHODS"),
            ("H2", "Experimental Design"),
            ("H3", "Control Groups"),
            ("H3", "Randomization"),
            ("H2", "Survey Research"),
            ("H2", "Statistical Analysis"),
            ("H3", "Descriptive Statistics"),
            ("H3", "Inferential Statistics"),
            ("H1", "PART III: QUALITATIVE METHODS"),
            ("H2", "Interviews"),
            ("H2", "Focus Groups"),
            ("H2", "Case Studies"),
            ("H1", "PART IV: MIXED METHODS"),
            ("H2", "Sequential Design"),
            ("H2", "Concurrent Design"),
            ("H1", "PART V: WRITING UP"),
        ],
    },
    {
        "title": "Smart City Infrastructure Plan",
        "sections": [
            ("H1", "Vision and Goals"),
            ("H1", "Transportation"),
            ("H2", "Public Transit"),
            ("H3", "Bus Network"),
            ("H3", "Metro System"),
            ("H2", "Cycling Infrastructure"),
            ("H2", "EV Charging Stations"),
            ("H1", "Digital Infrastructure"),
            ("H2", "Broadband Network"),
            ("H2", "IoT Sensors"),
            ("H3", "Traffic Sensors"),
            ("H3", "Environmental Sensors"),
            ("H1", "Energy"),
            ("H2", "Smart Grid"),
            ("H2", "District Heating"),
            ("H1", "Governance"),
            ("H2", "Data Platform"),
            ("H2", "Citizen Portal"),
            ("H1", "Implementation Timeline"),
        ],
    },
]


# ── Font configurations (variety across PDFs) ─────────────────────────────────

FONT_CONFIGS = [
    # Config 0: Standard academic
    {"title_size": 24, "h1_size": 18, "h2_size": 14, "h3_size": 12, "body_size": 10,
     "h1_bold": True, "h2_bold": True, "h3_bold": True, "font": "helv"},
    # Config 1: Larger headings
    {"title_size": 28, "h1_size": 20, "h2_size": 16, "h3_size": 13, "body_size": 11,
     "h1_bold": True, "h2_bold": True, "h3_bold": False, "font": "helv"},
    # Config 2: Tight spacing
    {"title_size": 22, "h1_size": 16, "h2_size": 13, "h3_size": 11, "body_size": 10,
     "h1_bold": True, "h2_bold": False, "h3_bold": False, "font": "times-roman"},
    # Config 3: Bold-heavy
    {"title_size": 26, "h1_size": 18, "h2_size": 15, "h3_size": 12, "body_size": 10,
     "h1_bold": True, "h2_bold": True, "h3_bold": True, "font": "courier"},
    # Config 4: Subtle hierarchy
    {"title_size": 20, "h1_size": 15, "h2_size": 13, "h3_size": 11.5, "body_size": 10.5,
     "h1_bold": True, "h2_bold": True, "h3_bold": False, "font": "helv"},
]


# ── Body text snippets ────────────────────────────────────────────────────────

BODY_PARAGRAPHS = [
    "This section provides an overview of the key concepts and methodologies discussed throughout the document. The analysis draws on both theoretical frameworks and empirical evidence gathered from multiple sources.",
    "Recent developments in this field have shown significant promise. Multiple research groups have independently verified the core findings, lending additional credibility to the proposed approach.",
    "The implementation details are described below. Each component was designed with modularity and extensibility in mind, following established software engineering best practices.",
    "Data was collected from a representative sample of participants over a six-month period. The sampling methodology ensured adequate representation across all relevant demographic categories.",
    "Performance metrics indicate substantial improvements over the baseline approach. The gains are statistically significant at the p < 0.01 level across all test conditions.",
    "Several limitations should be noted when interpreting these results. The sample size, while adequate for the primary analysis, may be insufficient for some of the secondary comparisons.",
    "Future work should focus on extending the methodology to additional domains and validating the generalizability of the current findings. Cross-cultural studies would be particularly valuable.",
    "The proposed framework offers a flexible and scalable solution that addresses the core challenges identified in the literature review. Its modular design allows for incremental adoption.",
    "Quality assurance procedures were implemented at each stage of the process. Regular audits and automated testing ensured compliance with the established standards and specifications.",
    "Stakeholder feedback was incorporated through a series of structured interviews and survey instruments. The results informed several key design decisions documented in this report.",
]


# ── PDF Generator ──────────────────────────────────────────────────────────────

def generate_pdf(
    template: dict,
    font_config: dict,
    output_pdf: Path,
    output_gt: Path,
) -> None:
    """
    Generate a single PDF from a template and save the matching ground truth.
    """
    doc = fitz.open()
    rng = random.Random(hash(template["title"]))  # deterministic per title

    # Page dimensions (A4)
    page_width, page_height = 595.28, 841.89
    margin_left = 72.0
    margin_right = page_width - 72.0
    margin_top = 72.0
    margin_bottom = page_height - 72.0
    line_spacing = 1.4

    def new_page():
        return doc.new_page(width=page_width, height=page_height)

    page = new_page()
    y = margin_top
    current_page_num = 1

    def advance_y(amount: float):
        nonlocal y, page, current_page_num
        y += amount
        if y > margin_bottom - 20:
            page = new_page()
            current_page_num += 1
            y = margin_top

    # ── Title ──────────────────────────────────────────────────────────────────
    title_text = template["title"]
    title_size = font_config["title_size"]
    font_name = font_config["font"]

    page.insert_text(
        fitz.Point(margin_left, y + title_size),
        title_text,
        fontname=font_name,
        fontsize=title_size,
        color=(0, 0, 0),
    )
    y += title_size * 2.5

    # ── Sections ───────────────────────────────────────────────────────────────
    outline_entries = []

    for level, heading_text in template["sections"]:
        # Pick font size and bold for this heading level
        if level == "H1":
            hsize = font_config["h1_size"]
            bold = font_config["h1_bold"]
            pre_space = hsize * 1.8
            indent = 0.0
        elif level == "H2":
            hsize = font_config["h2_size"]
            bold = font_config["h2_bold"]
            pre_space = hsize * 1.3
            indent = 18.0
        else:  # H3
            hsize = font_config["h3_size"]
            bold = font_config["h3_bold"]
            pre_space = hsize * 1.0
            indent = 36.0

        advance_y(pre_space)

        # Bold uses the bold variant name
        actual_font = font_name
        if bold and font_name == "helv":
            actual_font = "hebo"
        elif bold and font_name == "times-roman":
            actual_font = "tibo"
        elif bold and font_name == "courier":
            actual_font = "cobo"

        heading_page = current_page_num

        page.insert_text(
            fitz.Point(margin_left + indent, y + hsize),
            heading_text,
            fontname=actual_font,
            fontsize=hsize,
            color=(0, 0, 0),
        )

        outline_entries.append({
            "level": level,
            "text": heading_text,
            "page": heading_page,
        })

        y += hsize * line_spacing

        # Insert 1-3 body paragraphs after each heading
        n_paras = rng.randint(1, 3)
        body_size = font_config["body_size"]
        for _ in range(n_paras):
            para = rng.choice(BODY_PARAGRAPHS)
            # Wrap long text manually (approx 80 chars per line at body size)
            words = para.split()
            lines = []
            current_line = ""
            for w in words:
                test = f"{current_line} {w}".strip()
                if len(test) > 85:
                    lines.append(current_line)
                    current_line = w
                else:
                    current_line = test
            if current_line:
                lines.append(current_line)

            for line in lines:
                advance_y(body_size * line_spacing)
                page.insert_text(
                    fitz.Point(margin_left, y),
                    line,
                    fontname=font_name,
                    fontsize=body_size,
                    color=(0.15, 0.15, 0.15),
                )

            advance_y(body_size * 0.8)  # paragraph gap

    # Save PDF
    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    doc.save(str(output_pdf))
    doc.close()

    # Save ground truth
    gt = {
        "title": title_text,
        "outline": outline_entries,
    }
    output_gt.parent.mkdir(parents=True, exist_ok=True)
    with open(output_gt, "w", encoding="utf-8") as f:
        json.dump(gt, f, indent=2, ensure_ascii=False)


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Generate synthetic training PDFs with ground truth."
    )
    parser.add_argument(
        "--out", type=Path, default=Path("ml/training_data"),
        help="Output directory for PDFs and GT JSONs."
    )
    parser.add_argument(
        "--count", type=int, default=15,
        help="Number of PDFs to generate (max = number of templates)."
    )
    args = parser.parse_args()

    out_dir = args.out
    pdf_dir = out_dir / "pdfs"
    gt_dir = out_dir / "ground_truth"

    count = min(args.count, len(TEMPLATES))

    print(f"Generating {count} synthetic training PDFs...")
    for i in range(count):
        template = TEMPLATES[i]
        font_config = FONT_CONFIGS[i % len(FONT_CONFIGS)]
        pdf_path = pdf_dir / f"synthetic_{i+1:02d}.pdf"
        gt_path = gt_dir / f"synthetic_{i+1:02d}.json"

        generate_pdf(template, font_config, pdf_path, gt_path)
        n_headings = len(template["sections"])
        print(f"  [{i+1:2d}/{count}] {pdf_path.name}  "
              f"({n_headings} headings, font config {i % len(FONT_CONFIGS)})")

    print(f"\nDone. PDFs: {pdf_dir}/  Ground truth: {gt_dir}/")
    print(f"Total headings across all PDFs: "
          f"{sum(len(t['sections']) for t in TEMPLATES[:count])}")


if __name__ == "__main__":
    main()
