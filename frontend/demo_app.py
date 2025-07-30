import streamlit as st
import requests
import os
import base64
import traceback
from dotenv import load_dotenv
import re
from datetime import datetime
from pathlib import Path
import json
from streamlit_cookies_manager import CookieManager

# Must be first Streamlit command
st.set_page_config(
    page_title="Legal Medical Analyzer",
    layout="wide",
    page_icon="⚖️"
)

# Load environment variables
env_path = Path(__file__).resolve().parent.parent / "config.env"
load_dotenv(env_path)

# Initialize cookies
cookies = None

def initialize_cookies():
    global cookies
    if cookies is None:
        try:
            cookies = CookieManager()
        except:
            # Fallback for when cookies aren't available
            class SimpleCookieManager:
                def get(self, key, default=None):
                    return default
                def __setitem__(self, key, value):
                    pass
            cookies = SimpleCookieManager()

# Firebase REST API endpoints
FIREBASE_API_KEY = os.getenv("FIREBASE_API_KEY")
SIGN_IN_URL = f"https://identitytoolkit.googleapis.com/v1/accounts:signInWithPassword?key={FIREBASE_API_KEY}"
SIGN_UP_URL = f"https://identitytoolkit.googleapis.com/v1/accounts:signUp?key={FIREBASE_API_KEY}"
REFRESH_TOKEN_URL = f"https://securetoken.googleapis.com/v1/token?key={FIREBASE_API_KEY}"

# =====================================================
# Authentication Functions
# =====================================================

def refresh_id_token():
    """Refresh Firebase ID token using refresh token"""
    if not st.session_state.get('refresh_token'):
        return False
        
    data = {
        "grant_type": "refresh_token",
        "refresh_token": st.session_state.refresh_token
    }
    
    try:
        response = requests.post(REFRESH_TOKEN_URL, data=data)
        if response.status_code == 200:
            token_data = response.json()
            st.session_state.id_token = token_data.get('id_token')
            st.session_state.refresh_token = token_data.get('refresh_token')
            cookies["refresh_token"] = token_data.get('refresh_token')
            return True
    except:
        pass
    return False

def firebase_sign_in(email, password):
    """Authenticate with Firebase REST API"""
    data = {
        "email": email,
        "password": password,
        "returnSecureToken": True
    }
    response = requests.post(SIGN_IN_URL, json=data)
    if response.status_code == 200:
        return response.json()
    else:
        error_msg = response.json().get("error", {}).get("message", "Authentication failed")
        raise Exception(error_msg)

def firebase_sign_up(email, password):
    """Create new user with Firebase REST API"""
    if len(password) < 6:
        raise Exception("Password must be at least 6 characters")
        
    data = {
        "email": email,
        "password": password,
        "returnSecureToken": True
    }
    response = requests.post(SIGN_UP_URL, json=data)
    if response.status_code == 200:
        return response.json()
    else:
        error_msg = response.json().get("error", {}).get("message", "Account creation failed")
        raise Exception(error_msg)

def firebase_reset_password(email):
    """Send password reset email"""
    data = {
        "requestType": "PASSWORD_RESET",
        "email": email
    }
    response = requests.post(
        f"https://identitytoolkit.googleapis.com/v1/accounts:sendOobCode?key={FIREBASE_API_KEY}", 
        json=data
    )
    if response.status_code == 200:
        return True
    else:
        error_msg = response.json().get("error", {}).get("message", "Password reset failed")
        raise Exception(error_msg)

# =====================================================
# Document Processing Functions
# =====================================================

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

# =====================================================
# Session Management
# =====================================================

def initialize_session_state():
    initialize_cookies()

    # Ensure essential session state variables exist
    essential_vars = [
        'authenticated', 'user_email', 'user_uid', 'id_token', 
        'refresh_token', 'processed_files', 'errors', 'show_login',
        'show_signup', 'password_reset', 'selected_tool', 'chat_history',
        'missing_reports', 'history', 'selected_analysis'
    ]
    
    for var in essential_vars:
        if var not in st.session_state:
            # Set appropriate default values
            if var in ['authenticated', 'show_login', 'show_signup', 'password_reset']:
                st.session_state[var] = False
            elif var in ['user_email', 'user_uid', 'id_token', 'refresh_token']:
                st.session_state[var] = None
            elif var in ['processed_files', 'errors', 'chat_history', 'missing_reports']:
                st.session_state[var] = []
            elif var == 'history':
                st.session_state[var] = {}
            elif var == 'selected_tool':
                st.session_state[var] = "Document Analyzer"
            elif var == 'selected_analysis':
                st.session_state[var] = None
    
    # Special case: show_login should default to True
    if not st.session_state.show_login and not any([
        st.session_state.show_signup,
        st.session_state.password_reset,
        st.session_state.authenticated
    ]):
        st.session_state.show_login = True
    if 'authenticated' not in st.session_state:
        st.session_state.authenticated = False
    if 'user_email' not in st.session_state:
        st.session_state.user_email = None
    if 'user_uid' not in st.session_state:
        st.session_state.user_uid = None
    if 'id_token' not in st.session_state:
        st.session_state.id_token = None
    if 'refresh_token' not in st.session_state:
        st.session_state.refresh_token = None
    if 'processed_files' not in st.session_state:
        st.session_state.processed_files = []
    if 'errors' not in st.session_state:
        st.session_state.errors = []
    if 'show_login' not in st.session_state:
        st.session_state.show_login = True
    if 'show_signup' not in st.session_state:
        st.session_state.show_signup = False
    if 'password_reset' not in st.session_state:
        st.session_state.password_reset = False
    if 'selected_tool' not in st.session_state:
        st.session_state.selected_tool = "Document Analyzer"
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'missing_reports' not in st.session_state:
        st.session_state.missing_reports = []
    if 'history' not in st.session_state:
        st.session_state.history = {}
    if 'selected_analysis' not in st.session_state:
        st.session_state.selected_analysis = None

def handle_authentication():
    #initialize_cookies() 
    initialize_session_state()
    # Check for existing session via cookies
    refresh_token = None
    try:
        refresh_token = cookies.get("refresh_token")
    except:
        pass
    
    if not st.session_state.authenticated and refresh_token and refresh_token.strip():
        if not st.session_state.authenticated and cookies.get("refresh_token"):
            st.session_state.refresh_token = cookies.get("refresh_token")
            if refresh_id_token():
                st.session_state.authenticated = True
                # Simple token decoding (for demo only)
                try:
                    payload = st.session_state.id_token.split('.')[1]
                    payload += '=' * (4 - len(payload) % 4)  # Add padding
                    decoded = json.loads(base64.b64decode(payload).decode('utf-8'))
                    st.session_state.user_email = decoded.get('email')
                    st.session_state.user_uid = decoded.get('user_id')
                except:
                    pass
                return
    
    # Show auth UI if not authenticated
    if not st.session_state.authenticated:
        with st.container():
            st.markdown("## 🔒 Legal Case Analysis Platform")
            
            # Login Form
            if st.session_state.show_login:
                with st.form("login_form"):
                    email = st.text_input("Email")
                    password = st.text_input("Password", type="password")
                    submit = st.form_submit_button("Login")
                    
                    if submit:
                        try:
                            user_data = firebase_sign_in(email, password)
                            st.session_state.authenticated = True
                            st.session_state.user_email = email
                            st.session_state.id_token = user_data.get('idToken')
                            st.session_state.refresh_token = user_data.get('refreshToken')
                            cookies["refresh_token"] = user_data.get('refreshToken')
                            st.rerun()
                        except Exception as e:
                            st.error(f"Login failed: {str(e)}")
                
                col1, col2 = st.columns([1, 2])
                with col1:
                    if st.button("Create Account"):
                        st.session_state.show_login = False
                        st.session_state.show_signup = True
                        st.rerun()
                with col2:
                    if st.button("Reset Password"):
                        st.session_state.show_login = False
                        st.session_state.password_reset = True
                        st.rerun()
            
            # Signup Form
            elif st.session_state.show_signup:
                with st.form("signup_form"):
                    st.subheader("Create New Account")
                    new_email = st.text_input("Email")
                    new_password = st.text_input("Password", type="password")
                    confirm_password = st.text_input("Confirm Password", type="password")
                    submit = st.form_submit_button("Create Account")
                    
                    if submit:
                        if new_password != confirm_password:
                            st.error("Passwords do not match")
                        else:
                            try:
                                user_data = firebase_sign_up(new_email, new_password)
                                st.success("Account created successfully! Please login")
                                st.session_state.show_signup = False
                                st.session_state.show_login = True
                                st.rerun()
                            except Exception as e:
                                st.error(f"Signup failed: {str(e)}")
                
                if st.button("Back to Login"):
                    st.session_state.show_signup = False
                    st.session_state.show_login = True
                    st.rerun()
            
            # Password Reset
            elif st.session_state.password_reset:
                with st.form("reset_form"):
                    st.subheader("Password Reset")
                    reset_email = st.text_input("Enter your email")
                    submit = st.form_submit_button("Send Reset Link")
                    
                    if submit:
                        try:
                            firebase_reset_password(reset_email)
                            st.success("Password reset email sent! Check your inbox")
                            st.session_state.password_reset = False
                            st.session_state.show_login = True
                            st.rerun()
                        except Exception as e:
                            st.error(f"Reset failed: {str(e)}")
                
                if st.button("Back to Login"):
                    st.session_state.password_reset = False
                    st.session_state.show_login = True
                    st.rerun()
            
            st.stop()

# =====================================================
# Document Analysis Functions
# =====================================================

def validate_file(file):
    allowed = ['pdf', 'docx']
    return file.name.split('.')[-1].lower() in allowed

def process_files(files):
    st.session_state.processed_files = []
    st.session_state.errors = []
    st.session_state.missing_reports = []
    
    with st.status("Analyzing documents...", expanded=True) as status:
        try:
            prog_bar = st.progress(0)
            
            for idx, file in enumerate(files):
                try:
                    # Reset file pointer to beginning
                    file.seek(0)
                    
                    response = requests.post(
                        os.getenv("API_URL"),
                        files={"file": file},
                        headers={"X-API-Key": os.getenv("API_KEY")},
                        timeout=120
                    )
                    
                    if response.ok:
                        try:
                            result = response.json()
                            st.session_state.processed_files.append(result)
                            st.success(f"Processed: {file.name}")
                        except json.JSONDecodeError:
                            st.error(f"Invalid response for {file.name}")
                            st.session_state.errors.append({
                                "file": file.name,
                                "error": "Invalid JSON response from API",
                                "type": file.type,
                                "response": response.text[:500]
                            })
                    else:
                        error_text = response.text
                        try:
                            error_data = response.json()
                            error_details = error_data.get('details', error_text)
                        except:
                            error_details = error_text
                            
                        st.session_state.errors.append({
                            "file": file.name,
                            "error": error_details,
                            "type": file.type,
                            "trace": error_text
                        })
                        st.error(f"Failed: {file.name}")
                    
                except Exception as e:
                    st.session_state.errors.append({
                        "file": file.name,
                        "error": str(e),
                        "type": file.type,
                        "trace": traceback.format_exc()
                    })
                    st.error(f"Error: {file.name}")
                
                prog_bar.progress((idx+1)/len(files))
        
        finally:
            status.update(label="Analysis complete", state="complete")

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
            if st.session_state.chat_history and st.session_state.chat_history[-1]["role"] == "user":
                st.session_state.chat_history.pop()  # Remove failed query

# =====================================================
# Main Application Interfaces
# =====================================================

def document_analyzer_interface():
    # Try to load last analysis if session state is empty
    if not st.session_state.get('processed_files') and st.session_state.authenticated:
        try:
            response = requests.get(
                os.getenv("API_URL").replace('/process', '/get_last_analysis'),
                headers={
                    "X-API-Key": os.getenv("API_KEY"),
                    "X-User-UID": st.session_state.user_uid
                },
                params={"type": "document_analyses"}
            )
            
            if response.ok and response.json():
                last_analysis = response.json()[0]  # Get most recent
                st.session_state.processed_files = last_analysis.get('results', [])
                st.session_state.missing_reports = last_analysis.get('missing_reports', [])
                st.toast("Loaded your last document analysis")
        except Exception as e:
            st.error(f"Load error: {str(e)}")
    #mih dele?
    if st.session_state.selected_analysis:
        # Populate session state from saved analysis
        analysis = st.session_state.selected_analysis
        st.session_state.processed_files = analysis.get('results', [])
        st.session_state.missing_reports = analysis.get('missing_reports', [])
        st.session_state.selected_analysis = None  # Reset after loading
        
        st.success("Loaded saved analysis!")
        display_document_analysis_results()
        return
    st.subheader("Medical Document Analyzer", divider="green")
    
    # Upload section
    with st.expander("📁 Upload Documents", expanded=True):
        files = st.file_uploader(
            "Select files (PDF/DOCX)",
            type=["pdf", "docx"],
            accept_multiple_files=True,
            key="doc_analyzer_uploader"
        )
        
        process_btn = st.button("🚀 Start Analysis")
    
    # Processing logic
    if process_btn:
        if not files:
            st.warning("Please upload documents")
            return
        
        invalid = [f.name for f in files if not validate_file(f)]
        if invalid:
            st.error(f"Invalid files: {', '.join(invalid)}")
            return
        
        # Process files through API
        process_files(files)
        save_analysis_to_history("document_analyses", {
            "files": [f.name for f in files],
            "results": st.session_state.processed_files,
            "missing_reports": st.session_state.missing_reports
        })
    
    # Display results if available
    if st.session_state.processed_files:
        display_document_analysis_results()

    
    # Show any errors
    if hasattr(st.session_state, 'errors') and st.session_state.errors:
        with st.expander("⚠️ Processing Errors", expanded=False):
            for error in st.session_state.errors:
                st.error(f"**File:** {error['file']}")
                st.error(f"**Error:** {error['error']}")
                if 'trace' in error:
                    st.text_area("Traceback", error['trace'], height=100)

def display_document_analysis_results():
    """Display the actual document analysis results"""
    st.subheader("Analysis Results", divider="blue")
    
    # Merged chronology
    all_events = []
    for result in st.session_state.processed_files:
        if 'analysis' in result:
            try:
                events = parse_chronology(result['analysis'])
                all_events.extend(events)
            except Exception as e:
                st.error(f"Failed to parse chronology for {result.get('filename', 'unknown')}: {str(e)}")
    
    if all_events:
        with st.expander("📅 Integrated Medical Chronology", expanded=True):
            for event in sorted(all_events, key=lambda x: x['date']):
                st.markdown(f"**{event['date']}**: {event['description']}")
    
    # Individual document analysis
    st.subheader("Document Details", divider="green")
    for result in st.session_state.processed_files:
        with st.expander(f"📄 {result.get('filename', 'Unnamed file')}"):
            if 'processed_at' in result:
                st.write(f"**Processed:** {result['processed_at']}")
            
            if 'content' in result:
                st.text_area("Content Preview", result['content'], height=200, key=f"content_{result['filename']}")
            
            if 'analysis' in result:
                st.markdown("**AI Analysis**")
                st.markdown(result['analysis'])
    
    # Document completeness check
    with st.expander("🔍 Document Completeness Check", expanded=True):
        if 'missing_reports' not in st.session_state or not st.session_state.missing_reports:
            try:
                # Prepare the payload - send both full objects and analysis texts
                payload = {
                    "analyses": [
                        {
                            "object": res,  # The full result object
                            "text": res.get('analysis', '')  # Just the analysis text
                        }
                        for res in st.session_state.processed_files
                    ]
                }

                response = requests.post(
                    os.getenv("API_URL").replace('/process', '/detect_missing'),
                    json=payload,
                    headers={"X-API-Key": os.getenv("API_KEY")},
                    timeout=30
                )

                if response.ok:
                    st.session_state.missing_reports = response.json().get('missing', [])
                else:
                    st.error(f"Failed to check document completeness: {response.text}")
                    st.session_state.missing_reports = []
            except Exception as e:
                st.error(f"Completeness check error: {str(e)}")
                st.session_state.missing_reports = []
    
    # Add chat interface
    chat_interface()

def save_analysis_to_history(analysis_type, data):
    try:
        payload = {
            "type": analysis_type,
            "user_uid": st.session_state.user_uid,
            "user_email": st.session_state.user_email,
            "data": data
        }
        
        response = requests.post(
            os.getenv("API_URL").replace('/process', '/save_analysis'),
            json=payload,
            headers={
                "X-API-Key": os.getenv("API_KEY"),
                "X-User-UID": st.session_state.user_uid
            }
        )
        
        if response.status_code == 200:
            result = response.json()
            if result.get('status') == "success":
                return True
            else:
                st.error(f"Save skipped: {result.get('message', '')}")
        else:
            st.error(f"Save failed: {response.status_code} - {response.text}")
    except Exception as e:
        st.error(f"Save error: {str(e)}")
    return False
def malpractice_reviewer_interface():
    # Initialize save tracking
    if 'malpractice_saved' not in st.session_state:
        st.session_state.malpractice_saved = False
    
    # Try to load last analysis
    if not st.session_state.get('current_malpractice_analysis') and st.session_state.authenticated:
        try:
            response = requests.get(
                os.getenv("API_URL").replace('/process', '/get_last_analysis'),
                headers={
                    "X-API-Key": os.getenv("API_KEY"),
                    "X-User-UID": st.session_state.user_uid
                },
                params={"type": "malpractice_reviews"}
            )
            
            if response.status_code == 200 and response.json():
                last_analysis = response.json()[0]
                st.session_state.current_malpractice_analysis = {
                    'specialty': last_analysis.get('specialty', ''),
                    'content': last_analysis.get('content', ''),
                    'files': last_analysis.get('files', [])
                }
                st.session_state.malpractice_saved = True  # Mark as already saved
                st.toast("Loaded your last malpractice review")
        except Exception as e:
            st.error(f"Load error: {str(e)}")
    
    st.subheader("Medical Malpractice Reviewer", divider="red")
    
    # Specialty selection
    SPECIALTIES = [
        "Obstetrics", 
        "Emergency Medicine", 
        "Stroke/Neurology",
        "Oncology",
        "ENT",
        "General Surgery"
    ]
    
    # Set default specialty based on current analysis
    default_idx = 0
    if st.session_state.get('current_malpractice_analysis'):
        specialty_value = st.session_state.current_malpractice_analysis['specialty']
        if specialty_value in SPECIALTIES:
            default_idx = SPECIALTIES.index(specialty_value)
    
    specialty = st.selectbox("Select Medical Specialty", SPECIALTIES, index=default_idx)
    
    # Document upload
    st.subheader("Upload Medical Records")
    files = st.file_uploader(
        "Select medical records (PDF/DOCX)",
        type=["pdf", "docx"],
        accept_multiple_files=True,
        key="malpractice_uploader"
    )
    
    analyze_btn = st.button("Analyze Standard of Care Compliance")
    
    # Processing logic
    if analyze_btn:
        st.session_state.malpractice_saved = False  # Reset save flag
        if not files:
            st.warning("Please upload medical records first")
            return
            
        # Process files
        with st.status("Analyzing standard of care compliance...", expanded=True):
            st.write("Step 1: Extracting text from documents...")
            extracted_texts = []
            prog_bar = st.progress(0)
            
            for idx, file in enumerate(files):
                try:
                    response = requests.post(
                        os.getenv("API_URL").replace('/process', '/extract_text'),
                        files={"file": file},
                        headers={"X-API-Key": os.getenv("API_KEY")},
                        timeout=120
                    )
                    
                    if response.status_code == 200:
                        extracted_texts.append(response.json())
                    else:
                        st.error(f"Failed to process {file.name}: {response.text}")
                except Exception as e:
                    st.error(f"Error processing {file.name}: {str(e)}")
                
                prog_bar.progress((idx+1)/len(files))
            
            if not extracted_texts:
                st.error("No documents processed successfully")
                return
                
            st.write("Step 2: Analyzing standard of care and causation...")
            try:
                response = requests.post(
                    os.getenv('API_URL').replace('/process','/analyze_malpractice'),
                    json={
                        "specialty": specialty,
                        "documents": extracted_texts
                    },
                    headers={"X-API-Key": os.getenv("API_KEY")},
                    timeout=300  # 5 minutes
                )
                
                if response.status_code == 200:
                    analysis_data = response.json()
                    st.session_state.current_malpractice_analysis = {
                        'specialty': specialty,
                        'content': analysis_data['content'],
                        'files': [f['filename'] for f in extracted_texts]
                    }
                    st.success("Analysis complete!")
                else:
                    st.error(f"Analysis failed: {response.text}")
            except requests.exceptions.Timeout:
                st.error("Analysis timed out after 5 minutes. Please try again with fewer documents.")
                return
    
    # Display results if available and save only once
    if (st.session_state.get('current_malpractice_analysis') and 
        not st.session_state.malpractice_saved):
        
        analysis = st.session_state.current_malpractice_analysis
        st.subheader(f"Standard of Care Evaluation: {analysis['specialty']}", divider="green")
        
        # Display analysis content
        st.markdown(analysis['content'])
        
        # Add download button
        st.download_button(
            label="Download Report",
            data=analysis['content'],
            file_name=f"malpractice_review_{datetime.now().strftime('%Y%m%d')}.md",
            mime="text/markdown"
        )
        
        # Show file list
        with st.expander("Uploaded Documents"):
            for f in analysis['files']:
                st.write(f"- {f}")
        
        # Save to history
        if save_analysis_to_history("malpractice_reviews", {
            "specialty": analysis['specialty'],
            "content": analysis['content'],
            "files": analysis['files']
        }):
            st.session_state.malpractice_saved = True  # Mark as saved

def history_interface():
    st.subheader("Analysis History", divider="violet")
    st.info(f"Current User UID: {st.session_state.user_uid}")
    # Initialize analysis cache
    if 'analysis_cache' not in st.session_state:
        st.session_state.analysis_cache = {}
    
    # Load history
    try:
        with st.spinner("Loading recent history..."):
            response = requests.get(
                os.getenv("API_URL").replace('/process', '/get_history'),
                headers={
                    "X-API-Key": os.getenv("API_KEY"),
                    "X-User-UID": st.session_state.user_uid
                },
                timeout=30
            )
            
            if response.status_code == 200:
                history_data = response.json()
                # Remove duplicates by ID and sort by date
                seen_ids = set()
                unique_history = []
                for item in history_data:
                    if item['id'] not in seen_ids:
                        seen_ids.add(item['id'])
                        unique_history.append(item)
                
                # Sort by date descending and take only last 4 entries
                unique_history.sort(
                    key=lambda x: x.get('created_at', ''),
                    reverse=True
                )
                st.session_state.history = unique_history[:4]  # Only last 4 entries
                st.success(f"Loaded {len(st.session_state.history)} most recent history items")
            else:
                st.error(f"History error: {response.status_code} - {response.text}")
                st.session_state.history = []
    except Exception as e:
        st.error(f"Connection error: {str(e)}")
        st.session_state.history = []
    
    if not st.session_state.history:
        st.info("No analysis history found")
        st.write("Analyses will appear here after you process documents")
        return
    
    # Show history items
    for item in st.session_state.history:
        # Get actual files from results if available
        files = []
        if 'results' in item and item['results']:
            files = [r.get('filename', 'unnamed') for r in item['results']]
        elif 'files' in item:
            files = item['files']
        
        file_count = len(files)
        created_at = item.get('created_at', '')
        
        # Format timestamp for better readability
        try:
            dt = datetime.fromisoformat(created_at)
            display_date = dt.strftime("%b %d, %Y %H:%M")
        except:
            display_date = created_at  # Fallback if formatting fails
        
        # Use tool-specific icon and title
        analysis_type = item.get('type', 'Unknown')
        specialty = item.get('specialty', '')
        
        if analysis_type == 'document_analyses':
            icon = "📄"
            title = "Document Analysis"
        elif analysis_type == 'malpractice_reviews':
            icon = "⚕️"
            title = f"Malpractice Review: {specialty}"
        else:
            icon = "📝"
            title = "Analysis"
        
        # Display compact history item
        with st.container():
            col1, col2 = st.columns([4, 1])
            with col1:
                # Display icon and title with formatted date
                st.markdown(f"**{icon} {title}**")
                st.caption(f"{display_date} • {file_count} file{'s' if file_count != 1 else ''}")
                
                # Display file names in tooltip
                if files:
                    with st.expander(f"View {file_count} file{'s' if file_count != 1 else ''}"):
                        for f in files:
                            st.write(f"- {f}")
            
            with col2:
                # Load button with loading indicator
                if st.button("Load", key=f"load_{item['id']}"):
                    try:
                        # Load full analysis
                        with st.spinner("Loading analysis..."):
                            response = requests.get(
                                f"{os.getenv('API_URL').replace('/process', '/get_analysis')}",
                                headers={
                                    "X-API-Key": os.getenv("API_KEY"),
                                    "X-User-UID": st.session_state.user_uid
                                },
                                params={"id": item['id']},
                                timeout=30
                            )
                            
                            if response.status_code == 200:
                                full_data = response.json()
                                # Store in cache
                                st.session_state.analysis_cache[item['id']] = full_data
                                st.success("Analysis loaded!")
                                st.rerun()
                            else:
                                st.error(f"Failed to load analysis: {response.text}")
                    except Exception as e:
                        st.error(f"Load error: {str(e)}")
            
            # Display analysis if available in cache
            analysis_data = st.session_state.analysis_cache.get(item['id'])
            if analysis_data:
                st.subheader("Analysis Results", divider="gray")
                
                # Display based on analysis type
                if analysis_data.get('type') == 'document_analyses':
                    # Display each document's analysis
                    for result in analysis_data.get('results', []):
                        st.markdown(f"##### 📄 {result.get('filename', 'Unnamed file')}")
                        
                        if 'processed_at' in result:
                            st.caption(f"Processed: {result['processed_at']}")
                            
                        if 'content' in result:
                            with st.expander("Content Preview"):
                                st.text(result['content'])
                            
                        if 'analysis' in result:
                            with st.expander("AI Analysis"):
                                st.markdown(result['analysis'])
                    
                    # Show missing reports if available
                    if 'missing_reports' in analysis_data and analysis_data['missing_reports']:
                        st.subheader("Missing Documents", divider="gray")
                        st.write("The following document types appear to be missing:")
                        for doc in analysis_data['missing_reports']:
                            st.write(f"- {doc}")
                
                elif analysis_data.get('type') == 'malpractice_reviews':
                    st.markdown(analysis_data.get('content', ''))
                    if 'specialty' in analysis_data:
                        st.write(f"**Medical Specialty:** {analysis_data['specialty']}")
            
            # Add space between history items
            st.divider()

def clean_history(history):
    """Remove duplicate history entries"""
    seen = set()
    unique = []
    for item in history:
        # Create unique identifier
        identifier = f"{item.get('created_at','')}-{'-'.join(item.get('files',[]))}"
        
        if identifier not in seen:
            seen.add(identifier)
            unique.append(item)
    return unique

# =====================================================
# Main Application Structure
# =====================================================

def main_interface():
    # Tool selection in sidebar
    st.sidebar.title("Medical Legal Suite")
    
    if st.session_state.authenticated:
        st.sidebar.subheader(f"Welcome, {st.session_state.user_email}")
    
    # Tool selection
    TOOLS = {
        "Document Analyzer": "📄",
        "Malpractice Reviewer": "⚕️",
        "Analysis History": "🕒"
    }
    
    # Get current tool selection from session state
    selected_tool = st.sidebar.radio(
        "Select Tool", 
        list(TOOLS.keys()),
        format_func=lambda x: f"{TOOLS[x]} {x}",
        key="tool_selection_radio"  # Use a unique key for the widget
    )
    # Update session state with the selected tool
    st.session_state.selected_tool = selected_tool

    
    # Main content area
    st.title(f"{TOOLS[st.session_state.selected_tool]} {st.session_state.selected_tool}")
    
    if st.session_state.selected_tool == "Document Analyzer":
        document_analyzer_interface()
    elif st.session_state.selected_tool == "Malpractice Reviewer":
        malpractice_reviewer_interface()
    elif st.session_state.selected_tool == "Analysis History":
        history_interface()
        
    # Logout button
    if st.session_state.authenticated:
        st.sidebar.divider()
        if st.sidebar.button("Logout"):
            # Clear session state
            st.session_state.authenticated = False
            st.session_state.user_email = None
            st.session_state.user_uid = None
            st.session_state.id_token = None
            st.session_state.refresh_token = None
            st.session_state.processed_files = []
            st.session_state.errors = []
            
            # Clear cookies
            try:
                cookies["refresh_token"] = ""
                # Force cookie expiration
                cookies._save_cookie("refresh_token", "", max_age=0)
            except:
                pass
            
            # Clear chat history
            if 'chat_history' in st.session_state:
                del st.session_state['chat_history']
                
            st.success("You have been logged out successfully!")
            st.rerun()

# =====================================================
# Main Application Flow
# =====================================================

def main():
    initialize_session_state()
    handle_authentication()
    main_interface()

if __name__ == '__main__':
    main()
