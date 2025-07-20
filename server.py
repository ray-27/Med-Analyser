from flask import Flask, request, jsonify, send_from_directory, send_file, Response
# from werkzeug.utils import secure_filename 
import os
import uuid
import json
from datetime import datetime
from typing import Dict, List
import tempfile
import asyncio
from threading import Thread
from flask import render_template
import queue
import time
import requests 
from flask_cors import CORS
from dotenv import load_dotenv
# Import your existing tools
from file_processing import FileTools  # Your OCR class
from agent import MedicalAgent  # Your LangGraph agent

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes
load_dotenv()

# Configuration
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['PROCESSED_FOLDER'] = 'processed'
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB max file size
app.config['SECRET_KEY'] = 'your-secret-key'

# Allowed file extensions
ALLOWED_EXTENSIONS = {'pdf', 'png', 'jpg', 'jpeg', 'gif', 'bmp'}

# Create directories
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['PROCESSED_FOLDER'], exist_ok=True)

model_service = os.getenv('MODEL_SERVICE')
# Initialize tools
file_tools = FileTools()
medical_agent = MedicalAgent(api_type=model_service)  # Your LangGraph agent

# Storage for processed documents (in production, use a database)
document_storage = {}

def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def generate_session_id():
    """Generate unique session ID"""
    return str(uuid.uuid4())

def process_uploaded_files(files: List, session_id: str) -> Dict:
    """Process uploaded files and extract medical data"""
    processed_files = []
    all_medical_data = []
    
    for i,file in enumerate(files):
        try:
            # Save uploaded file
            filename = file.filename
            file_id = f"doc_{session_id}_{i+1}"
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{session_id}_{filename}")
            file.save(file_path)
            
            # Extract medical summary based on file type
            if filename.lower().endswith('.pdf'):
                result = file_tools.extract_pdf_prescription_summary(file_path)
            else:
                result = file_tools.extract_prescription_summary(file_path)
            
            if result["success"]:
                file_data = {
                    "file_id": file_id,
                    "original_name": filename,
                    "file_path": os.path.normpath(file_path),
                    "summary": result["summary"],
                    "method": result["method"],
                    "processed_at": datetime.now().isoformat(),
                    "file_type":  filename.split('.')[-1].lower() if '.' in filename else 'unknown'
                }
                processed_files.append(file_data)
                all_medical_data.append(result["summary"])
            else:
                # Handle failed processing
                processed_files.append({
                    "file_id": file_id,
                    "original_name": filename,
                    "error": result["error"],
                    "processed_at": datetime.now().isoformat()
                })
                
        except Exception as e:
            processed_files.append({
                "file_id": file_id if 'file_id' in locals() else "unknown",
                "original_name": filename,
                "error": str(e),
                "processed_at": datetime.now().isoformat()
            })
    
    return {
        "processed_files": processed_files,
        "combined_medical_data": "\n\n".join(all_medical_data),
        "session_id": session_id
    }

@app.route('/')
def index():
    """Serve the main frontend page"""
    return render_template('index.html')


@app.route('/view-document/<session_id>/<filename>')
def view_document(session_id, filename):
    """Serve original document files for viewing"""
    try:
        # Construct the file path
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        
        # Security check - ensure file exists and belongs to session
        if not os.path.exists(file_path) or session_id not in filename:
            return jsonify({'error': 'File not found'}), 404
        
        # Get file extension to determine content type
        file_ext = filename.lower().split('.')[-1]
        
        if file_ext == 'pdf':
            return send_file(file_path, mimetype='application/pdf')
        elif file_ext in ['jpg', 'jpeg', 'png', 'gif', 'bmp']:
            return send_file(file_path, mimetype=f'image/{file_ext}')
        else:
            return send_file(file_path)
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/how-to-use')
def how_to_use():
    """Serve the how-to-use instruction page"""
    return render_template('howtouse.html')

@app.route('/sample-files')
def sample_files():
    """Serve the sample files selection page"""
    return render_template('sample-files.html')

@app.route('/proxy-download/<filename>')
def proxy_download(filename):
    """Proxy download to force file download instead of display"""
    
    # Security check - only allow specific files
    file_mapping = {
        'IMG_2379.jpeg': 'https://dv8tzhzvv961l.cloudfront.net/build/IMG_2379.jpeg',
        'IMG_2380.jpeg': 'https://dv8tzhzvv961l.cloudfront.net/build/IMG_2380.jpeg',
        'IMG_2381.jpeg': 'https://dv8tzhzvv961l.cloudfront.net/build/IMG_2381.jpeg',
    }
    
    print(f"Download request for: {filename}")  # Debug log
    
    if filename not in file_mapping:
        print(f"File not found: {filename}")
        return jsonify({'error': 'File not found'}), 404
    
    try:
        url = file_mapping[filename]
        print(f"Fetching from: {url}")  # Debug log
        
        # Import requests here if global import fails
        import requests
        
        # Fetch file from CloudFront with timeout
        response = requests.get(url, timeout=30)
        print(f"CloudFront response status: {response.status_code}")  # Debug log
        
        if response.status_code == 200:
            print(f"File fetched successfully, size: {len(response.content)} bytes")
            
            # Return file with download headers
            return Response(
                response.content,
                mimetype='application/octet-stream',
                headers={
                    'Content-Disposition': f'attachment; filename="{filename}"',
                    'Content-Type': 'application/octet-stream',
                    'Content-Length': str(len(response.content))
                }
            )
        else:
            print(f"CloudFront returned status: {response.status_code}")
            return jsonify({'error': 'File not available from CloudFront'}), 404
            
    except ImportError as e:
        print(f"Import error: {e}")
        return jsonify({'error': 'Requests module not available. Please install: pip install requests'}), 500
        
    except requests.exceptions.RequestException as e:
        print(f"Request error: {e}")
        return jsonify({'error': f'Failed to fetch file: {str(e)}'}), 500
        
    except Exception as e:
        print(f"Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'Download error: {str(e)}'}), 500


@app.route('/download-samples', methods=['POST'])
def download_samples():
    """Force download of sample files from CloudFront"""
    try:
        sample_files = [
            ('IMG_2379.jpeg', 'https://dv8tzhzvv961l.cloudfront.net/build/IMG_2379.jpeg'),
            ('IMG_2380.jpeg', 'https://dv8tzhzvv961l.cloudfront.net/build/IMG_2380.jpeg'),
            ('IMG_2381.jpeg', 'https://dv8tzhzvv961l.cloudfront.net/build/IMG_2381.jpeg'),
        ]
        
        # Generate download script with forced download
        download_script = '''
        <script>
            console.log("Starting downloads...");
            
            async function downloadFile(url, filename, delay) {
                return new Promise((resolve) => {
                    setTimeout(async () => {
                        try {
                            // Fetch the file as blob to force download
                            const response = await fetch(url);
                            const blob = await response.blob();
                            
                            // Create object URL from blob
                            const objectUrl = URL.createObjectURL(blob);
                            
                            // Create download link
                            const link = document.createElement('a');
                            link.href = objectUrl;
                            link.download = filename;
                            link.style.display = 'none';
                            
                            // Trigger download
                            document.body.appendChild(link);
                            link.click();
                            document.body.removeChild(link);
                            
                            // Clean up object URL
                            URL.revokeObjectURL(objectUrl);
                            
                            console.log(`Downloaded: ${filename}`);
                            resolve();
                        } catch (error) {
                            console.error(`Failed to download ${filename}:`, error);
                            resolve();
                        }
                    }, delay);
                });
            }
        '''
        
        # Add download calls for each file
        for i, (filename, url) in enumerate(sample_files):
            delay = i * 1500  # 1.5 second delay between downloads
            download_script += f'''
            downloadFile('{url}', '{filename}', {delay});
            '''
        
        download_script += '''
        </script>
        '''
        
        # Add notification
        notification = f'''
        <div id="download-notification" style="position: fixed; top: 20px; right: 20px; background: #48bb78; color: white; padding: 15px 20px; border-radius: 8px; z-index: 9999; box-shadow: 0 4px 15px rgba(0,0,0,0.2);">
            <div style="display: flex; align-items: center; gap: 10px;">
                <span>📥</span>
                <div>
                    <strong>Downloading {len(sample_files)} Sample Files</strong>
                    <div style="font-size: 0.9em; opacity: 0.9;">Medical document samples</div>
                </div>
                <button onclick="this.parentElement.parentElement.remove()" 
                        style="background: none; border: none; color: white; cursor: pointer; font-size: 1.2em;">×</button>
            </div>
        </div>
        <script>
            // Auto remove notification after 8 seconds
            setTimeout(() => {{
                const notification = document.getElementById('download-notification');
                if (notification) notification.remove();
            }}, 8000);
        </script>
        '''
        
        return download_script + notification
        
    except Exception as e:
        return f'''
        <div style="position: fixed; top: 20px; right: 20px; background: #e53e3e; color: white; padding: 15px 20px; border-radius: 8px; z-index: 9999;">
            ❌ Download failed: {str(e)}
            <button onclick="this.parentElement.remove()" style="background: none; border: none; color: white; cursor: pointer; margin-left: 10px;">×</button>
        </div>
        '''


# @app.route('/v1/chat/completions', methods=['POST'])
# async def proxy_to_ollama():
#     """Convert vLLM format to Ollama format"""
#     vllm_request = request.json
    
#     # Convert vLLM format to Ollama format
#     ollama_payload = {
#         "model": vllm_request.get("model", "llama3.1:8b"),
#         "messages": vllm_request.get("messages", []),
#         "stream": False,
#         "options": {
#             "num_predict": vllm_request.get("max_tokens", 512),
#             "temperature": vllm_request.get("temperature", 0.7)
#         }
#     }
    
#     # Call Ollama
#     async with httpx.AsyncClient() as client:
#         response = await client.post(
#             "http://localhost:11434/api/chat",
#             json=ollama_payload
#         )
#         ollama_result = response.json()
    
#     # Convert back to vLLM format
#     vllm_response = {
#         "choices": [{
#             "message": {
#                 "content": ollama_result["message"]["content"]
#             }
#         }]
#     }
    
#     return jsonify(vllm_response)

@app.route('/static/<path:filename>')
def static_files(filename):
    """Serve static files (CSS, JS)"""
    return send_from_directory('templates', filename)

@app.route('/css')
def serve_css():
    return send_from_directory('templates', 'style.css', mimetype='text/css')

@app.route('/js')
def serve_js():
    return send_from_directory('templates', 'script.js', mimetype='application/javascript')

@app.route('/upload', methods=['POST'])
def upload_files():
    """
    Upload and process medical files
    Returns: Session ID and processed medical data
    """
    try:
        # Check if files are present
        if 'files' not in request.files:
            return jsonify({
                'error': 'No files provided',
                'status': 'error'
            }), 400
        
        files = request.files.getlist('files')
        
        if not files or files[0].filename == '':
            return jsonify({
                'error': 'No files selected',
                'status': 'error'
            }), 400
        
        # Validate file types
        invalid_files = []
        valid_files = []
        
        for file in files:
            if file and allowed_file(file.filename):
                valid_files.append(file)
            else:
                invalid_files.append(file.filename)
        
        if not valid_files:
            return jsonify({
                'error': f'No valid files. Invalid files: {invalid_files}',
                'status': 'error'
            }), 400
        
        # Generate session ID
        session_id = generate_session_id()
        
        # Process files
        try:
            processing_result = process_uploaded_files(valid_files, session_id)
        except Exception as process_error:
            print(f"File processing error: {process_error}")
            import traceback
            traceback.print_exc()
            return jsonify({
                'error': f'File processing failed: {str(process_error)}',
                'status': 'error'
            }), 500
        
        # Store session data
        try:
            document_storage[session_id] = {
                'files': processing_result['processed_files'],
                'medical_data': processing_result['combined_medical_data'],
                'created_at': datetime.now().isoformat(),
                'status': 'ready'
            }
        except Exception as storage_error:
            print(f"Session storage error: {storage_error}")
            return jsonify({
                'error': f'Session storage failed: {str(storage_error)}',
                'status': 'error'
            }), 500
        
        # Initialize LangGraph agent with the medical data
        try:
            medical_agent.initialize_session(
                session_id=session_id,
                processed_files=processing_result['processed_files']
            )
        except Exception as agent_error:
            print(f"Agent initialization error: {agent_error}")
            import traceback
            traceback.print_exc()
            return jsonify({
                'error': f'Agent initialization failed: {str(agent_error)}',
                'status': 'error'
            }), 500
        
        return jsonify({
            'session_id': session_id,
            'status': 'success',
            'message': 'Files processed successfully',
            'processed_files': len(processing_result['processed_files']),
            'medical_summary': processing_result['combined_medical_data'][:500] + "..." if len(processing_result['combined_medical_data']) > 500 else processing_result['combined_medical_data'],
            'invalid_files': invalid_files
        })
        
    except Exception as e:
        print(f"Upload route error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'error': str(e),
            'status': 'error'
        }), 500

@app.route('/ask', methods=['POST'])
def ask_question():
    """Ask questions about medical documents"""
    try:
        data = request.get_json()
        
        if not data or 'session_id' not in data or 'question' not in data:
            return jsonify({
                'error': 'session_id and question are required',
                'status': 'error'
            }), 400
            
        session_id = data['session_id']
        question = data['question']
        
        print(f"Processing question: {question} for session: {session_id}")
        
        # Check if session exists
        if session_id not in document_storage:
            return jsonify({
                'error': 'Invalid session_id or session expired',
                'status': 'error'
            }), 404
        
        # Process question
        try:
            result = medical_agent.process_question(session_id, question)
            print(f"Agent result keys: {list(result.keys())}")
            print(f"Agent answer length: {len(result.get('answer', ''))}")
            
            # Get filename mapping for referenced documents
            session_files = document_storage[session_id].get('files', [])
            referenced_filenames = []
            
            for doc_ref in result.get('document_references', []):
                file_info = next((f for f in session_files if f.get('file_id') == doc_ref), None)
                if file_info and file_info.get('original_name'):
                    referenced_filenames.append(file_info['original_name'])
                else:
                    referenced_filenames.append(doc_ref)
            
            # Ensure all required fields are present
            response_data = {
                'session_id': session_id,
                'question': question,
                'answer': str(result.get('answer', '')),  # Ensure it's a string
                'document_references': result.get('document_references', []),
                'referenced_filenames': referenced_filenames,
                'confidence_score': float(result.get('confidence_score', 0.0)),
                'sources_used': int(result.get('sources_used', 0)),
                'status': 'success'
            }
            
            print(f"Returning response: {response_data}")
            return jsonify(response_data)
            
        except Exception as agent_error:
            print(f"Agent error: {agent_error}")
            import traceback
            traceback.print_exc()
            
            return jsonify({
                'error': f'Agent processing error: {str(agent_error)}',
                'status': 'error'
            }), 500
            
    except Exception as e:
        print(f"Request error: {e}")
        import traceback
        traceback.print_exc()
        
        return jsonify({
            'error': str(e),
            'status': 'error'
        }), 500




# @app.route('/ask', methods=['POST'])
# def ask_question():
#     """Ask questions about medical documents (synchronous)"""
#     try:
#         data = request.get_json()
        
#         if not data or 'session_id' not in data or 'question' not in data:
#             return jsonify({
#                 'error': 'session_id and question are required',
#                 'status': 'error'
#             }), 400
            
#         session_id = data['session_id']
#         question = data['question']
        
#         print(f"Processing question: {question} for session: {session_id}")
        
#         # Check if session exists
#         if session_id not in document_storage:
#             return jsonify({
#                 'error': 'Invalid session_id or session expired',
#                 'status': 'error'
#             }), 404
        
#         # ✅ Call synchronously - no asyncio.run() needed
#         try:
#             result = medical_agent.process_question(session_id, question)
#             print(f"Agent result: {result}")
            
#             return jsonify({
#                 'session_id': session_id,
#                 'question': question,
#                 'answer': result.get('answer', 'No response generated'),
#                 'document_references': result.get('document_references', []),
#                 'confidence_score': result.get('confidence_score', 0.0),
#                 'sources_used': result.get('sources_used', 0),
#                 'status': 'success'
#             })
            
#         except Exception as agent_error:
#             print(f"Agent error: {agent_error}")
#             import traceback
#             traceback.print_exc()
            
#             return jsonify({
#                 'error': f'Agent processing error: {str(agent_error)}',
#                 'status': 'error'
#             }), 500
            
#     except Exception as e:
#         print(f"Request error: {e}")
#         return jsonify({
#             'error': str(e),
#             'status': 'error'
#         }), 500



@app.route('/session/<session_id>', methods=['GET'])
def get_session_info(session_id):
    """Get information about a session"""
    if session_id not in document_storage:
        return jsonify({
            'error': 'Session not found',
            'status': 'error'
        }), 404
    
    session_data = document_storage[session_id]
    
    # Ensure files is properly structured
    files = session_data.get('files', [])
    
    return jsonify({
        'session_id': session_id,
        'status': session_data.get('status', 'ready'),
        'created_at': session_data.get('created_at', ''),
        'files': files,  # Make sure this key exists
        'files_count': len(files),
        'medical_summary': session_data.get('medical_data', '')[:500],
        'qa_count': len(session_data.get('qa_history', []))
    })


@app.route('/session/<session_id>/history', methods=['GET'])
def get_qa_history(session_id):
    """Get Q&A history for a session"""
    if session_id not in document_storage:
        return jsonify({
            'error': 'Session not found',
            'status': 'error'
        }), 404
    
    return jsonify({
        'session_id': session_id,
        'qa_history': document_storage[session_id].get('qa_history', []),
        'status': 'success'
    })

@app.route('/session/<session_id>', methods=['DELETE'])
def delete_session(session_id):
    """Delete a session and cleanup files"""
    if session_id not in document_storage:
        return jsonify({
            'error': 'Session not found',
            'status': 'error'
        }), 404
    
    # Cleanup files
    session_data = document_storage[session_id]
    for file_info in session_data['files']:
        try:
            if os.path.exists(file_info['file_path']):
                os.remove(file_info['file_path'])
        except Exception as e:
            print(f"Error deleting file {file_info['file_path']}: {e}")
    
    # Remove from storage
    del document_storage[session_id]
    
    # Cleanup agent session
    medical_agent.cleanup_session(session_id)
    
    return jsonify({
        'message': 'Session deleted successfully',
        'status': 'success'
    })

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'active_sessions': len(document_storage),
        'timestamp': datetime.now().isoformat()
    })

# Add to the end of your Flask app
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8080)