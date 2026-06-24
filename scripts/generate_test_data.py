import json
import os
import random
from pathlib import Path

try:
    from reportlab.pdfgen import canvas
    from reportlab.lib.pagesizes import letter
    from reportlab.pdfbase.ttfonts import TTFont
    from reportlab.pdfbase import pdfmetrics
except ImportError:
    print("reportlab not installed. Please run: pip install reportlab")
    exit(1)

def generate_lorem_ipsum(words=50):
    text = (
        "Lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. "
        "Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. "
        "Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. "
        "Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum. "
        "Curabitur pretium tincidunt lacus. Nulla gravida orci a odio. Nullam varius, turpis et commodo pharetra, est eros "
        "bibendum elit, nec luctus magna felis sollicitudin mauris. Integer in mauris eu nibh euismod gravida. Duis ac tellus "
        "et risus vulputate vehicula. Donec lobortis risus a elit. Etiam tempor. Ut ullamcorper, ligula eu tempor congue, eros "
        "est euismod turpis, id tincidunt sapien risus a quam. Maecenas fermentum consequat mi. Donec fermentum. Pellentesque "
        "malesuada nulla a mi. Duis sapien sem, aliquet nec, commodo eget, consequat quis, neque. Aliquam faucibus, elit ut "
        "dictum aliquet, felis nisl adipiscing sapien, sed malesuada diam lacus eget erat."
    )
    tokens = text.split()
    if words < len(tokens):
        return " ".join(tokens[:words]) + "."
    return " ".join(tokens * (words // len(tokens) + 1))[:words*7] + "..."

def get_font_families():
    return [
        {"regular": "Helvetica", "bold": "Helvetica-Bold", "italic": "Helvetica-Oblique"},
        {"regular": "Times-Roman", "bold": "Times-Bold", "italic": "Times-Italic"},
        {"regular": "Courier", "bold": "Courier-Bold", "italic": "Courier-Oblique"}
    ]

def create_document(pdf_path, json_path):
    c = canvas.Canvas(str(pdf_path), pagesize=letter)
    width, height = letter
    
    font_family = random.choice(get_font_families())
    base_font = font_family["regular"]
    bold_font = font_family["bold"]
    
    # Randomly decide sizing logic for this document
    h1_size = random.choice([18, 20, 22, 24])
    h2_size = random.choice([14, 15, 16])
    h3_size = random.choice([12, 13])
    body_size = random.choice([10, 11])
    
    # Document structure
    num_h1 = random.randint(1, 4)
    y_pos = height - 50
    x_margin = 50
    
    headings_truth = []
    
    def check_page_break(current_y, needed_space=50):
        nonlocal c, y_pos
        if current_y < needed_space:
            c.showPage()
            y_pos = height - 50
            return True
        return False

    def draw_wrapped_text(text, x, y, font, size, max_width, line_height=1.2):
        nonlocal c, y_pos
        c.setFont(font, size)
        words = text.split()
        line = []
        for word in words:
            if c.stringWidth(" ".join(line + [word]), font, size) < max_width:
                line.append(word)
            else:
                c.drawString(x, y_pos, " ".join(line))
                y_pos -= (size * line_height)
                check_page_break(y_pos)
                c.setFont(font, size)
                line = [word]
        if line:
            c.drawString(x, y_pos, " ".join(line))
            y_pos -= (size * line_height)
    
    h1_counter = 0
    h2_counter = 0
    h3_counter = 0
    
    use_numbering = random.choice([True, False])

    for i in range(num_h1):
        check_page_break(y_pos, h1_size + body_size + 20)
        h1_counter += 1
        h2_counter = 0
        
        prefix = f"{h1_counter}. " if use_numbering else ""
        text = f"{prefix}Main Section {h1_counter}: {generate_lorem_ipsum(random.randint(2, 5)).strip('.')}"
        
        # Center H1 sometimes
        is_centered = random.choice([True, False])
        c.setFont(bold_font, h1_size)
        text_width = c.stringWidth(text, bold_font, h1_size)
        draw_x = (width - text_width) / 2 if is_centered else x_margin
        
        c.drawString(draw_x, y_pos, text)
        headings_truth.append({"level": "H1", "text": text})
        
        y_pos -= (h1_size + 15)
        
        # Paragraphs
        for _ in range(random.randint(1, 2)):
            check_page_break(y_pos)
            draw_wrapped_text(generate_lorem_ipsum(random.randint(20, 60)), x_margin, y_pos, base_font, body_size, width - 100)
            y_pos -= 10
            
        num_h2 = random.randint(1, 3)
        for j in range(num_h2):
            check_page_break(y_pos, h2_size + body_size + 20)
            h2_counter += 1
            h3_counter = 0
            
            prefix_h2 = f"{h1_counter}.{h2_counter} " if use_numbering else ""
            text_h2 = f"{prefix_h2}Subsection {h2_counter}: {generate_lorem_ipsum(random.randint(3, 6)).strip('.')}"
            
            c.setFont(bold_font, h2_size)
            c.drawString(x_margin, y_pos, text_h2)
            headings_truth.append({"level": "H2", "text": text_h2})
            
            y_pos -= (h2_size + 12)
            
            for _ in range(random.randint(1, 2)):
                check_page_break(y_pos)
                draw_wrapped_text(generate_lorem_ipsum(random.randint(20, 50)), x_margin, y_pos, base_font, body_size, width - 100)
                y_pos -= 10
                
            num_h3 = random.randint(0, 2)
            for k in range(num_h3):
                check_page_break(y_pos, h3_size + body_size + 20)
                h3_counter += 1
                
                prefix_h3 = f"{h1_counter}.{h2_counter}.{h3_counter} " if use_numbering else ""
                text_h3 = f"{prefix_h3}Sub-subsection {h3_counter}"
                
                # Sometime indent H3
                indent = random.choice([0, 20])
                c.setFont(bold_font, h3_size)
                c.drawString(x_margin + indent, y_pos, text_h3)
                headings_truth.append({"level": "H3", "text": text_h3})
                
                y_pos -= (h3_size + 10)
                
                for _ in range(random.randint(1, 2)):
                    check_page_break(y_pos)
                    draw_wrapped_text(generate_lorem_ipsum(random.randint(15, 30)), x_margin + indent, y_pos, base_font, body_size, width - 100 - indent)
                    y_pos -= 10
                    
    c.save()
    
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(headings_truth, f, indent=2)

def main():
    base_dir = Path(__file__).resolve().parent.parent
    pdfs_dir = base_dir / "sample_datasets" / "pdfs_test"
    gt_dir = base_dir / "sample_datasets" / "ground_truth"
    
    pdfs_dir.mkdir(parents=True, exist_ok=True)
    gt_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Generating 100 PDFs into {pdfs_dir}")
    print(f"Generating 100 Ground Truth JSONs into {gt_dir}")
    
    for i in range(1, 101):
        pdf_path = pdfs_dir / f"test_doc_{i:03d}.pdf"
        json_path = gt_dir / f"test_doc_{i:03d}.json"
        
        create_document(pdf_path, json_path)
        
    print("Generation complete.")

if __name__ == "__main__":
    main()
