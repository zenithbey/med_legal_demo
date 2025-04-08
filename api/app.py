from flask import Flask, request, jsonify
from document_processor import process_document
import os
import traceback
from pathlib import Path
from dotenv import load_dotenv
import magic
from openai import OpenAIError

env_path = Path(__file__).resolve().parent.parent / "config.env"
print(f"Looking for config at: {env_path}")  # Debug line
load_dotenv(env_path)


app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB limit

def simple_auth(f):
    def wrapper(*args, **kwargs):
        if request.headers.get('X-API-Key') != os.getenv('API_KEY'):
            return jsonify({"error": "Unauthorized"}), 401
        return f(*args, **kwargs)
    wrapper.__name__ = f.__name__
    return wrapper

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

if __name__ == '__main__':
    print("API_KEY:", os.getenv('API_KEY'))
    app.run(host='0.0.0.0', port=5000, debug=True)