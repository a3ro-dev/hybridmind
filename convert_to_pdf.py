import os
import re
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT

def generate_pdf(md_path, pdf_path):
    with open(md_path, 'r', encoding='utf-8') as f:
        content = f.read()

    doc = SimpleDocTemplate(
        pdf_path,
        pagesize=letter,
        rightMargin=40,
        leftMargin=40,
        topMargin=40,
        bottomMargin=40
    )

    styles = getSampleStyleSheet()

    # Custom styles
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=22,
        leading=26,
        textColor=colors.HexColor('#1E293B'),
        spaceAfter=15
    )

    h2_style = ParagraphStyle(
        'DocH2',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=14,
        leading=18,
        textColor=colors.HexColor('#0F172A'),
        spaceBefore=14,
        spaceAfter=8,
        keepWithNext=True
    )

    body_style = ParagraphStyle(
        'DocBody',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=10,
        leading=14,
        textColor=colors.HexColor('#334155'),
        spaceAfter=8
    )

    bullet_style = ParagraphStyle(
        'DocBullet',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=10,
        leading=14,
        textColor=colors.HexColor('#334155'),
        leftIndent=15,
        spaceAfter=4
    )

    table_header_style = ParagraphStyle(
        'TableHeader',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=9,
        leading=11,
        textColor=colors.white
    )

    table_cell_style = ParagraphStyle(
        'TableCell',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=8.5,
        leading=11,
        textColor=colors.HexColor('#1E293B')
    )

    story = []

    # Parse markdown line by line/block by block
    lines = content.split('\n')
    in_mermaid = False
    in_table = False
    table_rows = []

    def format_text(text):
        # bold
        text = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', text)
        # italic
        text = re.sub(r'\*(.*?)\*', r'<i>\1</i>', text)
        return text

    i = 0
    while i < len(lines):
        line = lines[i].strip()

        if line.startswith('```mermaid'):
            in_mermaid = True
            i += 1
            continue
        elif in_mermaid:
            if line.startswith('```'):
                in_mermaid = False
                # Add visual roadmap representation block
                roadmap_text = """
                <b>HybridMind Execution Roadmap (6-Month Plan)</b><br/><br/>
                <b>Phase 1: Engine & Architecture (Aug - Sep 2026)</b><br/>
                • Core Tri-Signal Retrieval Engine & Atomic .mind Persistence<br/><br/>
                <b>Phase 2: Reranking & Beta Readiness (Oct 2026)</b><br/>
                • Graph Auto-Edges, GNN Embeddings & Cross-Encoder Reranking<br/><br/>
                <b>Phase 3: Beta Launch (Nov 2026)</b><br/>
                • <b>★ Beta Release & Open-Source Launch (Nov 2026)</b><br/><br/>
                <b>Phase 4: Pilots & v1.0 Release (Nov 2026 - Jan 2027)</b><br/>
                • Early Adopter Trials (3-5 AI Teams) & Seed Round Closing
                """
                box_style = ParagraphStyle('RoadmapBox', parent=body_style, fontSize=9.5, leading=14, textColor=colors.HexColor('#0F172A'))
                box_table = Table([[Paragraph(roadmap_text, box_style)]], colWidths=[532])
                box_table.setStyle(TableStyle([
                    ('BACKGROUND', (0,0), (-1,-1), colors.HexColor('#F8FAFC')),
                    ('BOX', (0,0), (-1,-1), 1, colors.HexColor('#CBD5E1')),
                    ('PADDING', (0,0), (-1,-1), 10),
                    ('BOTTOMPADDING', (0,0), (-1,-1), 12),
                ]))
                story.append(Spacer(1, 4))
                story.append(box_table)
                story.append(Spacer(1, 8))
            i += 1
            continue

        if line.startswith('|'):
            # Table row
            cells = [c.strip() for c in line.split('|')[1:-1]]
            if len(cells) > 0 and not all(set(c).issubset({'-', ':', ' '}) for c in cells):
                table_rows.append(cells)
            i += 1
            # Check if next line is table
            if i < len(lines) and lines[i].strip().startswith('|'):
                continue
            else:
                # Render accumulated table
                if table_rows:
                    header = table_rows[0]
                    body_rows = table_rows[1:]
                    
                    formatted_data = []
                    formatted_data.append([Paragraph(format_text(h), table_header_style) for h in header])
                    for r in body_rows:
                        formatted_data.append([Paragraph(format_text(cell), table_cell_style) for cell in r])

                    num_cols = len(header)
                    if num_cols == 6: # Feature comparison table
                        col_widths = [75, 100, 80, 85, 95, 97]
                    elif num_cols == 4: # Budget table
                        col_widths = [110, 70, 55, 297]
                    else:
                        col_widths = [532 / num_cols] * num_cols

                    t = Table(formatted_data, colWidths=col_widths)
                    t.setStyle(TableStyle([
                        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#1E293B')),
                        ('ALIGN', (0,0), (-1,-1), 'LEFT'),
                        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
                        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#E2E8F0')),
                        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor('#F8FAFC')]),
                        ('TOPPADDING', (0,0), (-1,-1), 5),
                        ('BOTTOMPADDING', (0,0), (-1,-1), 5),
                    ]))
                    story.append(Spacer(1, 4))
                    story.append(t)
                    story.append(Spacer(1, 8))
                    table_rows = []
            continue

        if line.startswith('# '):
            title_text = format_text(line[2:])
            story.append(Paragraph(title_text, title_style))
            story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor('#3B82F6'), spaceAfter=12))
        elif line.startswith('## '):
            h2_text = format_text(line[3:])
            story.append(Paragraph(h2_text, h2_style))
        elif line.startswith('- ') or line.startswith('* '):
            bullet_text = format_text(line[2:])
            story.append(Paragraph(f"• {bullet_text}", bullet_style))
        elif line != '':
            story.append(Paragraph(format_text(line), body_style))
        
        i += 1

    doc.build(story)
    print(f"Successfully generated PDF at: {pdf_path}")

if __name__ == '__main__':
    generate_pdf("deep-research-report (1).md", "deep-research-report.pdf")
