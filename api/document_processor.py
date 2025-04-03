import fitz
from pdf2image import convert_from_bytes
import pytesseract
from docx import Document
import io
from openai import OpenAI
import os
from pathlib import Path
from dotenv import load_dotenv
from datetime import datetime
import magic
import re

env_path = Path(__file__).resolve().parent.parent / "config.env"
load_dotenv(env_path)

client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

REQUIRED_DOC_TYPES = [
    "incident_report", 
    "medical_history", 
    "treatment_plan",
    "discharge_summary"
]

def extract_pdf_text(file_stream):
    """Extract text from searchable PDFs"""
    text = ""
    with fitz.open(stream=file_stream.read(), filetype="pdf") as doc:
        for page in doc:
            text += page.get_text()
    return text if text.strip() else None

def is_scanned_pdf(content):
    """Check if PDF needs OCR"""
    with fitz.open(stream=content, filetype="pdf") as doc:
        return not any(page.get_text().strip() for page in doc)

def ocr_pdf(file_stream):
    """Process scanned PDFs with OCR"""
    images = convert_from_bytes(file_stream.read())
    return "\n".join(pytesseract.image_to_string(img) for img in images)

def extract_docx_text(file_stream):
    """Extract text from DOCX files"""
    try:
        doc = Document(io.BytesIO(file_stream.read()))
        full_text = [para.text for para in doc.paragraphs]
        for table in doc.tables:
            full_text.extend(' | '.join(cell.text for cell in row.cells) 
                          for row in table.rows)
        return '\n'.join(filter(None, full_text))
    except Exception as e:
        raise ValueError(f"DOCX processing failed: {str(e)}")

def process_document(file, true_mime):
    """Main document processing function"""
    content = file.read()
    file.stream.seek(0)
    filename = (file.filename or "unnamed").lower()

    try:
        # PDF handling
        if true_mime == 'application/pdf':
            if is_scanned_pdf(content):
                text = ocr_pdf(io.BytesIO(content))
            else:
                text = extract_pdf_text(io.BytesIO(content))
        
        # DOCX handling
        elif true_mime == 'application/vnd.openxmlformats-officedocument.wordprocessingml.document':
            text = extract_docx_text(io.BytesIO(content))
        
        else:
            raise ValueError(f"Unsupported file type: {true_mime}")

        if not text.strip():
            raise ValueError("Document appears empty or unreadable")

        return {
            "filename": filename,
            "content": text[:1000] + "...",
            "analysis": analyze_with_gpt4(text),
            "mime_type": true_mime,
            "processed_at": datetime.now().isoformat()
        }

    except Exception as e:
        return {
            "filename": filename,
            "error": str(e),
            "status": "failed"
        }


def parse_chronology(analysis_text):
    """Convert AI output to structured timeline"""
    events = []
    date_pattern = r"\d{4}-\d{2}-\d{2}"
    
    for line in analysis_text.split('\n'):
        if "Date:" in line and "Event:" in line:
            date_match = re.search(date_pattern, line)
            if date_match:
                event = {
                    'date': datetime.strptime(date_match.group(), "%Y-%m-%d").date(),
                    'description': line.split("Event:")[-1].strip()
                }
                events.append(event)
    return sorted(events, key=lambda x: x['date'])

def analyze_with_gpt4(text):
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "system",
                "content": """Analyze medical documents and extract:
                1. Chronological events (date in YYYY-MM-DD format, description)
                2. Document type classification
                3. Missing required document types
                4. Potential liability factors. Use markdown formatting."""
            },
            {
                "role": "user",
                "content": text[:15000]  # Limit context size
            }
        ]
    )
    return response.choices[0].message.content

def detect_missing_reports(analyses):
    """Use GPT-4 to identify missing document types"""
    required_docs = [
        "Accident/Incident Report (official police or employer report)",
        "Hospital Admission Records (initial emergency room documentation)",
        "Discharge Summary (final hospital treatment summary)",
        "Imaging Results (X-ray, MRI, CT scan reports)",
        "Treatment Plans (ongoing care documentation)",
        "Insurance Claim Forms (submitted insurance documentation)"
    ]
    
    analysis_text = "\n\n".join([res['analysis'] for res in analyses])
    
    prompt = f"""Analyze these legal medical documents and identify if there is any missing documents:
    
    Required Document Types:
    {chr(10).join(required_docs)}
    
    Document Analysis:
    {analysis_text[:12000]}
    
    Return ONLY a comma-separated list of missing document type names from the required list.
    """
    
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a legal document analyst identifying missing case documents"},
            {"role": "user", "content": prompt}
        ],
        temperature=0.1
    )
    
    missing = response.choices[0].message.content.split(", ")
    return [doc for doc in required_docs if any(keyword in doc.lower() for keyword in missing)]

__all__ = ['process_document', 'detect_missing_reports'] 