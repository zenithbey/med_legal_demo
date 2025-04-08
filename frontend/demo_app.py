import streamlit as st
import requests
import os
import traceback
from dotenv import load_dotenv
import re
from datetime import datetime
import sys
from pathlib import Path
env_path = Path(__file__).resolve().parent.parent / "config.env"
print(f"Looking for config at: {env_path}")  # Debug line
load_dotenv(env_path)

def parse_chronology(analysis_text):
    """Extract dates and events from AI analysis output"""
    events = []
    try:
        # Look for markdown-style date headers
        date_matches = re.findall(r'\*\*(\d{4}-\d{2}-\d{2})\*\*: (.+)', analysis_text)
        for date_str, description in date_matches:
            events.append({
                'date': datetime.strptime(date_str, "%Y-%m-%d").date(),
                'description': description.strip()
            })
            
        # Fallback to any date-like patterns
        if not events:
            date_pattern = r"\b\d{4}-\d{2}-\d{2}\b"
            for line in analysis_text.split('\n'):
                if re.search(date_pattern, line):
                    date_match = re.search(date_pattern, line)
                    desc = line.replace(date_match.group(), '').strip()
                    events.append({
                        'date': datetime.strptime(date_match.group(), "%Y-%m-%d").date(),
                        'description': desc
                    })
                    
    except Exception as e:
        st.error(f"Chronology parsing error: {str(e)}")
        
    return events

def detect_missing_reports(processed_files):
    """Simplified frontend version without fitz"""
    return []

st.set_page_config(
    page_title="Legal Medical Analyzer",
    layout="wide",
    page_icon="⚖️"
)

def initialize_session_state():
    if 'authenticated' not in st.session_state:
        st.session_state.authenticated = False
    if 'processed_files' not in st.session_state:
        st.session_state.processed_files = []
    if 'errors' not in st.session_state:
        st.session_state.errors = []

def handle_authentication():
    if not st.session_state.authenticated:
        with st.container():
            st.markdown("## 🔒 Legal Case Analysis Platform")
            password = st.text_input("Enter Access Code", type="password")
            if st.button("Authenticate"):
                if password == os.getenv("DEMO_PASSWORD"):
                    st.session_state.authenticated = True
                    st.rerun()
                else:
                    st.error("Invalid access code")
            st.stop()

def validate_file(file):
    allowed = ['pdf', 'docx']
    return file.name.split('.')[-1].lower() in allowed

def process_files(files):
    st.session_state.processed_files = []
    st.session_state.errors = []
    
    # Status container moved outside any expanders
    with st.status("Analyzing documents...", expanded=True) as status:
        try:
            prog_bar = st.progress(0)
            
            for idx, file in enumerate(files):
                try:
                    response = requests.post(
                        os.getenv("API_URL"),
                        files={"file": file},
                        headers={"X-API-Key": os.getenv("API_KEY")},
                        timeout=45
                    )
                    
                    if response.ok:
                        st.session_state.processed_files.append(response.json())
                        st.success(f"Processed: {file.name}")
                    else:
                        error_data = response.json()
                        st.session_state.errors.append({
                            "file": file.name,
                            "error": error_data.get('details'),
                            "type": error_data.get('detected_type'),
                            "trace": error_data
                        })
                        st.error(f"Failed: {file.name}")
                    
                    prog_bar.progress((idx+1)/len(files))
                
                except Exception as e:
                    st.session_state.errors.append({
                        "file": file.name,
                        "error": str(e),
                        "type": file.type,
                        "trace": traceback.format_exc()
                    })
                    st.error(f"Error: {file.name}")
        
        finally:
            status.update(label="Analysis complete", state="complete")

def display_results():
    if st.session_state.processed_files:
        st.subheader("Analysis Results", divider="green")
        
        # Merged chronology
        all_events = []
        for result in st.session_state.processed_files:
            if 'analysis' not in result:  # Add safety check
                st.error(f"Missing analysis for {result.get('filename', 'unknown file')}")
                continue
                
            try:
                events = parse_chronology(result['analysis'])
                all_events.extend(events)
            except Exception as e:
                st.error(f"Failed to parse chronology for {result['filename']}: {str(e)}")
        
        with st.expander("📅 Integrated Medical Chronology", expanded=True):
            for event in sorted(all_events, key=lambda x: x['date']):
                st.markdown(f"**{event['date']}**: {event['description']}")

        # Individual document analysis
        st.subheader("Document Details", divider="blue")
        for result in st.session_state.processed_files:
            with st.expander(f"📄 {result['filename']}"):
                st.write(f"**Processed:** {result['processed_at']}")
                st.text_area("Content Preview", result['content'], height=200)
                st.markdown("**AI Analysis**")
                st.write(result['analysis'])
        with st.expander("🔍 Document Completeness Check", expanded=True):
            if 'missing_reports' not in st.session_state:
                with st.spinner("Analyzing document requirements..."):
                    try:
                        st.session_state.missing_reports = detect_missing_reports(
                            st.session_state.processed_files  # Pass full objects
                        )
                    except Exception as e:
                        st.error(f"Document analysis failed: {str(e)}")
                        st.session_state.missing_reports = []
            
            if st.session_state.missing_reports:
                st.error("Missing critical documents:")
                for doc in st.session_state.missing_reports:
                    st.write(f"- {doc}")
            else:
                st.success("All required documents present!")
                
        # Add chat interface
        chat_interface()                  
                
def main_interface():
    st.title("Medical Case Analysis Platform")
    
    # Upload section in separate expander
    with st.expander("📁 Upload Documents", expanded=True):
        files = st.file_uploader(
            "Select files (PDF/DOCX)",
            type=["pdf", "docx"],
            accept_multiple_files=True
        )
        
        process_btn = st.button("🚀 Start Analysis")
    
    # Processing and results outside upload expander
    if process_btn:
        if not files:
            st.warning("Please upload documents")
            return
        
        invalid = [f.name for f in files if not validate_file(f)]
        if invalid:
            st.error(f"Invalid files: {', '.join(invalid)}")
            return
        
        process_files(files)
    
    display_results()

def chat_interface():
    st.subheader("AI Legal Assistant", divider="blue")
    
    # Initialize chat history if not exists
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    
    # Display existing messages
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.write(message["content"])
    
    # Handle new input
    if prompt := st.chat_input("Ask about damages or case strategy"):
        try:
            # Add user message
            st.session_state.chat_history.append({"role": "user", "content": prompt})
            
            # Generate response with full context
            context = "\n".join([
                f"Document: {res['filename']}\nAnalysis: {res.get('analysis', '')}"
                for res in st.session_state.processed_files
            ])
            
            response = generate_chat_response(prompt, context)
            
            # Add AI response
            st.session_state.chat_history.append({"role": "assistant", "content": response})
            
            # Force rerun to update UI
            st.rerun()
            
        except Exception as e:
            st.error(f"Chat error: {str(e)}")
            st.session_state.chat_history.pop()  # Remove failed query

def generate_chat_response(prompt, context):
    from openai import OpenAI
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {
                "role": "system",
                "content": f"""Analyze these legal documents:
                {context[:12000]}
                
                Focus on:
                - Medical damages calculation
                - Liability assessment
                - Missing document implications"""
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0.3
    )
    
    return response.choices[0].message.content
def main():
    load_dotenv('../config.env')
    print("API_KEY:", os.getenv('API_KEY'))
    initialize_session_state()
    handle_authentication()
    main_interface()

if __name__ == '__main__':
    main()