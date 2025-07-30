from flask import Flask, request, jsonify
from document_processor import process_document, detect_missing_reports, extract_document_text, analyze_standard_of_care
import os
import traceback
from pathlib import Path
from dotenv import load_dotenv
import magic
from openai import OpenAIError
import firebase_admin
from firebase_admin import credentials, firestore
from google.cloud.firestore_v1 import FieldFilter
from flask_cors import CORS

env_path = Path(__file__).resolve().parent.parent / "config.env"
load_dotenv(env_path)

# Initialize Firebase with service account file
if not firebase_admin._apps:
    try:
        # Try to load from file in api directory
        cred_path = Path(__file__).resolve().parent / "firebase-service-account.json"
        cred = credentials.Certificate(str(cred_path))
        firebase_admin.initialize_app(cred)
    except Exception as e:
        print(f"Firebase init error: {str(e)}")
        # Fallback to environment variables if needed
        firebase_admin.initialize_app(options={'projectId': os.getenv("FIREBASE_PROJECT_ID")})

db = firestore.client()

app = Flask(__name__)
CORS(app)
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB limit

def simple_auth(f):
    def wrapper(*args, **kwargs):
        if request.headers.get('X-API-Key') != os.getenv('API_KEY'):
            return jsonify({"error": "Unauthorized"}), 401
        return f(*args, **kwargs)
    wrapper.__name__ = f.__name__
    return wrapper

@app.route('/detect_missing', methods=['POST'])
@simple_auth
def detect_missing():
    try:
        data = request.json
        
        # Expecting either a list of analysis texts or a list of analysis objects
        analysis_texts = []
        for item in data.get('analyses', []):
            if isinstance(item, dict):
                analysis_texts.append(item.get('analysis', ''))
            else:
                analysis_texts.append(str(item))

        missing = detect_missing_reports(analysis_texts)
        
        return jsonify({
            "status": "success",
            "missing": missing
        })
    
    except Exception as e:
        traceback.print_exc()
        return jsonify({
            "error": "processing_error",
            "details": str(e)
        }), 400

@app.route('/process', methods=['POST'])
@simple_auth
def process_file():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    
    file = request.files['file']
    try:
        header = file.stream.read(1024)
        file.stream.seek(0)
        
        mime = magic.Magic(mime=True)
        true_type = mime.from_buffer(header)
        
        # Special handling for DOCX/ZIP confusion
        if (true_type == 'application/zip' and 
            file.filename.lower().endswith('.docx')):
            true_type = 'application/vnd.openxmlformats-officedocument.wordprocessingml.document'
            
        result = process_document(file, true_type)
        return jsonify(result)
        
    except OpenAIError as oe:
        return jsonify({
            "error": "openai_error",
            "details": str(oe),
            "type": "llm_error"
        }), 500
    
    except Exception as e:
        traceback.print_exc()
        return jsonify({
            "error": "processing_error",
            "details": str(e),
            "filename": file.filename,
            "reported_type": file.content_type,
            "detected_type": true_type
        }), 400

# History management endpoints
@app.route('/save_analysis', methods=['POST'])
@simple_auth
def save_analysis():
    try:
        # Moved data assignment to the top
        data = request.json
        user_uid = data.get('user_uid')
        user_email = data.get('user_email')
        analysis_type = data.get('type')
        analysis_data = data.get('data', {})
        
        # Don't save empty analyses
        if not analysis_data.get('results') and not analysis_data.get('content'):
            return jsonify({
                "status": "skipped",
                "message": "Empty analysis not saved"
            }), 200
        
        if not user_uid or not analysis_type:
            return jsonify({"error": "Missing parameters"}), 400
        
        # Create document data
        doc_data = {
            "type": analysis_type,
            "user_uid": user_uid,
            "user_email": user_email,
            "created_at": firestore.SERVER_TIMESTAMP
        }
        
        # Add analysis-specific data
        if analysis_type == "document_analyses":
            doc_data["files"] = analysis_data.get('files', [])
            doc_data["results"] = analysis_data.get('results', [])
            doc_data["missing_reports"] = analysis_data.get('missing_reports', [])
        elif analysis_type == "malpractice_reviews":
            if not analysis_data.get('content'):
                return jsonify({
                    "status": "skipped",
                    "message": "Empty malpractice review not saved"
                }), 200
            doc_data["specialty"] = analysis_data.get('specialty', '')
            doc_data["content"] = analysis_data.get('content', '')
            doc_data["files"] = analysis_data.get('files', [])
        
        # Save to Firestore
        doc_ref = db.collection('analyses').document()
        doc_ref.set(doc_data)
        
        return jsonify({
            "status": "success",
            "doc_id": doc_ref.id
        })
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/delete_analysis', methods=['POST'])
@simple_auth
def delete_analysis():
    try:
        analysis_id = request.json.get('id')
        user_uid = request.json.get('user_uid')
        
        # Verify ownership before deleting
        doc = db.collection('analyses').document(analysis_id).get()
        if not doc.exists or doc.to_dict().get('user_uid') != user_uid:
            return jsonify({"error": "Unauthorized"}), 403
        
        db.collection('analyses').document(analysis_id).delete()
        return jsonify({"status": "success"})
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/get_last_analysis', methods=['GET'])
@simple_auth
def get_last_analysis():
    try:
        user_uid = request.headers.get('X-User-UID')
        analysis_type = request.args.get('type', 'document_analyses')
        
        if not user_uid:
            return jsonify({"error": "User UID required"}), 400
        
        # Create filters
        user_filter = FieldFilter("user_uid", "==", user_uid)
        type_filter = FieldFilter("type", "==", analysis_type)
        
        # Get last analysis
        docs = db.collection('analyses')\
                 .where(filter=user_filter)\
                 .where(filter=type_filter)\
                 .order_by('created_at', direction=firestore.Query.DESCENDING)\
                 .limit(1)\
                 .stream()
        
        history = []
        for doc in docs:
            data = doc.to_dict()
            data['id'] = doc.id
            # Convert timestamp
            if 'created_at' in data:
                data['created_at'] = data['created_at'].isoformat()
            history.append(data)
        
        return jsonify(history)
    
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route('/get_history', methods=['GET'])
@simple_auth
def get_history():
    try:
        user_uid = request.headers.get('X-User-UID')
        
        if not user_uid:
            return jsonify({"error": "User UID required"}), 400
        
        # Create query with proper ordering
        docs = db.collection('analyses')\
                 .where('user_uid', '==', user_uid)\
                 .order_by('created_at', direction=firestore.Query.DESCENDING)\
                 .get()
        
        history = []
        for doc in docs:
            data = doc.to_dict()
            data['id'] = doc.id
            # Convert timestamp to string
            if 'created_at' in data:
                # Handle both Timestamp and string formats
                if hasattr(data['created_at'], 'isoformat'):
                    data['created_at'] = data['created_at'].isoformat()
                elif isinstance(data['created_at'], str):
                    # Already string, keep as is
                    pass
            history.append(data)
        
        return jsonify(history)
    
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route('/get_analysis', methods=['GET'])
@simple_auth
def get_analysis():
    try:
        analysis_id = request.args.get('id')
        doc_ref = db.collection('analyses').document(analysis_id)
        doc = doc_ref.get()
        
        if not doc.exists:
            return jsonify({"error": "Analysis not found"}), 404
            
        data = doc.to_dict()
        data['id'] = doc.id
        
        # Ensure consistent files field
        if 'files' not in data or not data['files']:
            if 'results' in data and data['results']:
                data['files'] = [r.get('filename', 'unnamed') for r in data['results']]
            else:
                data['files'] = []
        
        # Convert timestamp
        if 'created_at' in data:
            data['created_at'] = data['created_at'].isoformat()
        
        return jsonify(data)
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/debug_history', methods=['GET'])
@simple_auth
def debug_history():
    try:
        user_uid = request.headers.get('X-User-UID')
        docs = db.collection('analyses')\
                 .where('user_uid', '==', user_uid)\
                 .stream()
        
        history = []
        for doc in docs:
            data = doc.to_dict()
            data['id'] = doc.id
            history.append(data)
        
        return jsonify({
            "count": len(history),
            "history": history
        })
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# New endpoint for malpractice analysis
@app.route('/analyze_malpractice', methods=['POST'])
@simple_auth
def analyze_malpractice():
    try:
        data = request.json
        specialty = data.get('specialty')
        documents = data.get('documents', [])
        
        if not specialty or not documents:
            return jsonify({"error": "Missing parameters"}), 400
            
        # Combine document texts (with filename context)
        combined_text = ""
        for doc in documents:
            combined_text += f"\n\nDocument: {doc.get('filename', 'unnamed')}\n"
            combined_text += doc.get('content', '')[:5000]  # Limit per document
            
        result = analyze_standard_of_care(specialty, combined_text[:15000])
        
        return jsonify({
            "specialty": specialty,
            "content": result,
            "documents": [d['filename'] for d in documents],
            "status": "success"
        })
    
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

# New endpoint for text extraction only
@app.route('/extract_text', methods=['POST'])
@simple_auth
def extract_text():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    
    file = request.files['file']
    try:
        header = file.stream.read(1024)
        file.stream.seek(0)
        
        mime = magic.Magic(mime=True)
        true_type = mime.from_buffer(header)
        
        # Special handling for DOCX/ZIP confusion
        if (true_type == 'application/zip' and 
            file.filename.lower().endswith('.docx')):
            true_type = 'application/vnd.openxmlformats-officedocument.wordprocessingml.document'
            
        # Use modified process_document that returns full text
        result = extract_document_text(file, true_type)
        return jsonify({
            "filename": file.filename,
            "content": result,
            "mime_type": true_type,
            "status": "success"
        })
        
    except Exception as e:
        traceback.print_exc()
        return jsonify({
            "error": "processing_error",
            "details": str(e),
            "filename": file.filename,
            "reported_type": file.content_type,
            "detected_type": true_type
        }), 400

if __name__ == '__main__':
    print("API_KEY:", os.getenv('API_KEY'))
    app.run(host='0.0.0.0', port=5000, debug=True)
