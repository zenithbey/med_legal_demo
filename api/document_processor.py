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
    text = ""
    with fitz.open(stream=file_stream.read(), filetype="pdf") as doc:
        for page in doc:
            text += page.get_text()
    return text if text.strip() else None

def ocr_pdf(file_stream):
    images = convert_from_bytes(file_stream.read())
    text = ""
    for img in images:
        text += pytesseract.image_to_string(img)
    return text

def extract_docx_text(file_stream):
    try:
        doc = Document(io.BytesIO(file_stream.read()))
        full_text = []
        
        for para in doc.paragraphs:
            full_text.append(para.text)
            
        for table in doc.tables:
            for row in table.rows:
                full_text.append(' | '.join(cell.text for cell in row.cells))
                
        return '\n'.join(full_text)
    except Exception as e:
        raise ValueError(f"DOCX processing failed: {str(e)}")

def process_document(file, true_mime):
    content = file.read()
    file.stream.seek(0)
    
    filename = (file.filename or "unnamed").lower()
    
    # Handle Office Open XML format detection
    if true_mime == 'application/zip' and filename.endswith('.docx'):
        true_mime = 'application/vnd.openxmlformats-officedocument.wordprocessingml.document'

    valid_types = {
        'application/pdf': extract_pdf_text,
        'application/vnd.openxmlformats-officedocument.wordprocessingml.document': extract_docx_text
    }
    
    if true_mime not in valid_types:
        raise ValueError(f"Unsupported file type: {true_mime} (Filename: {filename})")
    
    # Extract text
    text = valid_types[true_mime](io.BytesIO(content))
    
    if not text.strip():
        raise ValueError("Document appears empty or unreadable")
    
    # AI Analysis
    analysis = analyze_with_gpt4(text)
    
    return {
        "filename": filename,
        "content": text[:1000] + "...",
        "analysis": analysis,
        "mime_type": true_mime,
        "processed_at": datetime.now().isoformat()
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

def detect_missing_reports(processed_files):
    """GPT-4 powered missing document detection"""
    from openai import OpenAI
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    
    required_docs = [
        "Accident/Incident Report",
        "Hospital Admission Records",
        "Discharge Summary",
        "Imaging Results",
        "Treatment Plans",
        "Insurance Claim Forms"
    ]

    # Extract analysis texts safely
    analysis_texts = []
    for doc in processed_files:
        if isinstance(doc, dict) and 'analysis' in doc:
            analysis_texts.append(doc['analysis'])
        else:
            analysis_texts.append(str(doc))

    prompt = f"""Analyze these legal document analyses and list missing required docs:
    Required documents: {required_docs}
    
    Analyses:
    {' '.join(analysis_texts)[:12000]}
    
    Return ONLY missing document names as a comma-separated list.
    """

    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "Identify missing legal documents"},
            {"role": "user", "content": prompt}
        ],
        temperature=0.1
    )
    
    return [doc for doc in required_docs if doc.lower() in response.choices[0].message.content.lower()]

__all__ = ['process_document', 'detect_missing_reports'] 