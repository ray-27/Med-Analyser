import httpx
from typing import Dict, Any, List, Optional, TypedDict
from langgraph.graph import StateGraph, END, START
import json
import google.generativeai as genai
from datetime import datetime
import os
from dotenv import load_dotenv

import logging

logging.basicConfig(
    level=logging.INFO,  # Or use DEBUG for more detail
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler("med_agent.log", mode='a'),
        logging.StreamHandler()  # Keep this if you also want to see logs in the terminal
    ]
)
logging.getLogger('werkzeug').setLevel(logging.ERROR)


class MedicalAgentState(TypedDict):
    """Enhanced state for medical agent with memory and document tracking"""
    session_id: str
    current_question: str
    conversation_history: List[Dict]
    medical_context: Dict[str, Any]  # Document ID -> content mapping
    document_sources: Dict[str, Any]  # Document metadata
    extracted_context: List[Dict]  # Context with source references
    response: str
    document_references: List[str]  # Referenced document IDs
    confidence_score: float
    needs_clarification: bool
    error_message: str
    tool_calls: List[Dict]
    tool_results: List[Dict]
    question_analysis: Dict[str, Any]


class MedicalAgent:
    def __init__(self, 
                 model_endpoint: str = None,
                 model_name: str = None,
                 api_type: str = None):
        """Initialize agent with flexible model support (synchronous)"""
        load_dotenv()

        
        
        # Auto-detect API type based on environment or parameters
        self.gemini_api_key = os.getenv('GEMINI_API_KEY')
        self.ollama_endpoint = os.getenv('OLLAMA_ENDPOINT', 'http://localhost:11434')
        self.vllm_endpoint = os.getenv('VLLM_ENDPOINT', 'http://localhost:8000/v1')
        
        # Determine which API to use
        if api_type == 'ollama' or (not self.gemini_api_key and not model_endpoint):
            self.api_type = 'ollama'
            self.endpoint = self.ollama_endpoint
            self.model_name = model_name or os.getenv('OLLAMA_MODEL', 'llama3-groq-tool-use:8b')
        elif api_type == 'gemini' or self.gemini_api_key:
            self.api_type = 'gemini'
            self.setup_gemini()
        else:
            self.api_type = 'vllm'
            self.endpoint = model_endpoint or self.vllm_endpoint
            self.model_name = model_name or os.getenv('VLLM_MODEL', 'google/medgemma-4b-it')
        
        # Use synchronous HTTP client
        self.client = httpx.Client(timeout=30.0)

        self.sessions = {}
        
        # Initialize tools (no workflow needed for simplified version)
        self.tools = self._initialize_tools()
        
        # Medical knowledge base for validation
        self.medical_keywords = [
            "medication", "dosage", "prescription", "diagnosis", "symptoms",
            "treatment", "condition", "patient", "doctor", "follow-up"
        ]
        
        # Document storage with source tracking
        self.document_storage = {}
        logging.info(f"Initialized with {self.api_type} - Model: {self.model_name}")
    
    def setup_gemini(self):
        """Setup Gemini API"""
        genai.configure(api_key=self.gemini_api_key)
        self.model_name = os.getenv('GEMINI_AGENT_MODEL', 'gemini-2.0-flash-exp')
        self.gemini_model = genai.GenerativeModel(self.model_name)
    
    def _call_model(self, prompt: str, max_tokens: int = 512, temperature: float = 0.7) -> str:
        """Call the appropriate model API (synchronous)"""
        if self.api_type == 'ollama':
            return self._call_ollama_model(prompt, max_tokens, temperature)
        elif self.api_type == 'gemini':
            return self._call_gemini_model(prompt, max_tokens, temperature)
        else:
            return self._call_vllm_model(prompt, max_tokens, temperature)
    
    def _call_ollama_model(self, prompt: str, max_tokens: int = 512, temperature: float = 0.7) -> str:
        """Call Ollama API synchronously"""
        payload = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "options": {
                "num_predict": max_tokens,
                "temperature": temperature
            }
        }
        
        try:
            response = self.client.post(
                f"{self.endpoint}/api/chat",
                json=payload
            )
            response.raise_for_status()
            
            result = response.json()
            return result["message"]["content"]
            
        except httpx.HTTPError as e:
            raise Exception(f"Ollama API error: {str(e)}")

    def _call_gemini_model(self, prompt: str, max_tokens: int = 512, temperature: float = 0.7) -> str:
        """Call the Gemini API model synchronously"""
        try:
            # System prompt for medical context
            system_prompt = """You are a medical AI assistant that helps healthcare professionals analyze patient medical documents and history.

                    Your role:
                    - Analyze medical documents including prescriptions, diagnoses, lab results
                    - Extract relevant medical information accurately
                    - Provide clear, factual responses based on document content
                    - Always reference source documents when citing information

                    When responding:
                    - Be precise and factual
                    - Include document references when applicable
                    - If unsure, state limitations clearly"""
                                
            full_prompt = f"{system_prompt}\n\n{prompt}"
            
            # Configure generation parameters
            generation_config = genai.types.GenerationConfig(
                max_output_tokens=max_tokens,
                temperature=temperature
            )
            
            # Synchronous call to Gemini (no asyncio.to_thread needed)
            response = self.gemini_model.generate_content(
                full_prompt,
                generation_config=generation_config
            )
            
            return response.text
            
        except Exception as e:
            raise Exception(f"Gemini API error: {str(e)}")

    def _call_vllm_model(self, prompt: str, max_tokens: int = 512, temperature: float = 0.7) -> str:
        """Call the vLLM-hosted model synchronously"""
        payload = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": False
        }
        
        try:
            response = self.client.post(
                f"{self.vllm_endpoint}/chat/completions",
                json=payload
            )
            response.raise_for_status()
            
            result = response.json()
            return result["choices"][0]["message"]["content"]
            
        except httpx.HTTPError as e:
            raise Exception(f"vLLM API error: {str(e)}")
    
    def _initialize_tools(self):
        """Initialize available tools for the agent"""
        return {
            "search_documents": self._search_documents,
            "get_medication_info": self._get_medication_info,
            "calculate_dosage": self._calculate_dosage,
            "check_drug_interactions": self._check_drug_interactions,
            "get_document_summary": self._get_document_summary
        }

    # Simplified synchronous processing (no complex workflow)
    def process_question(self, session_id: str, question: str) -> Dict[str, Any]:
        """Process question synchronously with enhanced document referencing"""
        if session_id not in self.sessions:
            raise ValueError("Session not found. Please upload medical documents first.")
        
        session_data = self.sessions[session_id]
        
        try:
            logging.info(f"Processing question: {question} for session: {session_id}")
            
            # Get medical context and document sources
            medical_context = session_data.get('medical_context', {})
            document_sources = session_data.get('document_sources', {})
            conversation_history = session_data.get('conversation_history', [])
            
            logging.info(f"Medical context available: {len(medical_context)} documents")
            
            if not medical_context:
                return {
                    "answer": "I apologize, but I don't have access to any medical documents for this session. Please upload documents first.",
                    "document_references": [],
                    "confidence_score": 0.0,
                    "sources_used": 0
                }
            
            # Format medical context with original filenames
            context_text = ""
            for doc_id, content in medical_context.items():
                original_name = document_sources.get(doc_id, {}).get("original_name", doc_id)
                context_text += f"\nDocument {doc_id} ({original_name}):\n{content}\n"
            
            # Format conversation history
            history_text = ""
            if conversation_history:
                recent_history = conversation_history[-2:]
                for entry in recent_history:
                    history_text += f"User: {entry['question']}\nAssistant: {entry['answer']}\n\n"
            
            # Enhanced prompt for better document referencing
            prompt = f"""You are a medical AI assistant analyzing patient medical documents.

            Medical Documents Available:
            {context_text}

            Previous Conversation:
            {history_text}

            Current Question: {question}

            Based on the medical documents provided above, please answer the question accurately. 

            IMPORTANT: 
            - Do not include phrases like "According to document..." or "Based on file..."

            Answer:"""
                    
                    # Call model synchronously
            response = self._call_model(prompt, max_tokens=2000, temperature=0.7)
            
            # Parse response to extract only actually referenced documents
            referenced_docs = self._extract_referenced_documents(response, document_sources)
            
            # Store conversation
            conversation_entry = {
                "question": question,
                "answer": response,
                "timestamp": datetime.now().isoformat(),
                "document_references": referenced_docs
            }
            
            session_data['conversation_history'].append(conversation_entry)
            
            # Keep only last 10 conversations
            if len(session_data['conversation_history']) > 10:
                session_data['conversation_history'] = session_data['conversation_history'][-10:]
            
            return {
                "answer": response.strip(),
                "document_references": referenced_docs,  # Only referenced docs
                "confidence_score": 0.8,
                "sources_used": len(referenced_docs)
            }
            
        except Exception as e:
            logging.info(f"Error processing question: {e}")
            return {
                "answer": f"I apologize, but I encountered an error: {str(e)}",
                "document_references": [],
                "confidence_score": 0.0,
                "sources_used": 0
            }


    def initialize_session(self, session_id: str, processed_files: List[Dict]):
        """Initialize session with processed medical documents"""
        logging.info(f"Initializing session {session_id} with {len(processed_files)} files")
        
        medical_context = {}
        document_sources = {}
        
        for i, file_data in enumerate(processed_files):
            doc_id = f"doc_{session_id}_{i + 1}"
            
            summary = file_data.get("summary", "")
            logging.info(f"Document {doc_id}: {len(summary)} characters")
            
            medical_context[doc_id] = summary
            document_sources[doc_id] = {
                "original_name": file_data.get("original_name", ""),
                "file_type": file_data.get("file_type", ""),
                "processed_at": file_data.get("processed_at", ""),
                "method": file_data.get("method", ""),
                "file_path": file_data.get("file_path", "")
            }
        
        self.sessions[session_id] = {
            "medical_context": medical_context,
            "document_sources": document_sources,
            "conversation_history": [],
            "created_at": datetime.now().isoformat()
        }
        
        logging.info(f"Session initialized with medical_context keys: {list(medical_context.keys())}")

    def _extract_referenced_documents(self, response: str, document_sources: Dict) -> List[str]:
        """Extract document IDs that are actually referenced in the response"""
        referenced_docs = []
        
        # Look for filename patterns in the response
        for doc_id, source_info in document_sources.items():
            original_name = source_info.get("original_name", "")
            
            # Check if the original filename is mentioned in the response
            if original_name and original_name.lower() in response.lower():
                referenced_docs.append(doc_id)
                logging.info(f"Found reference to document: {original_name} (ID: {doc_id})")
            
            # Also check for document ID patterns (as fallback)
            if doc_id in response:
                referenced_docs.append(doc_id)
                logging.info(f"Found reference to document ID: {doc_id}")
        
        logging.info(f"Total referenced documents: {len(referenced_docs)}")
        return list(set(referenced_docs))  # Remove duplicates

    # Tool implementations (simplified, synchronous)
    def _search_documents(self, medical_context: Dict, query: str) -> List[Dict]:
        """Search through medical documents synchronously"""
        search_results = []
        
        for doc_id, content in medical_context.items():
            if query.lower() in content.lower():
                search_results.append({
                    "document_id": doc_id,
                    "content": content,
                    "relevance_score": self._calculate_relevance(query, content)
                })
        
        # Sort by relevance
        search_results.sort(key=lambda x: x["relevance_score"], reverse=True)
        return search_results[:5]  # Top 5 results
    
    def _calculate_relevance(self, query: str, content: str) -> float:
        """Calculate relevance score between query and content"""
        query_words = set(query.lower().split())
        content_words = set(content.lower().split())
        
        intersection = query_words.intersection(content_words)
        union = query_words.union(content_words)
        
        return len(intersection) / len(union) if union else 0
    
    def _get_medication_info(self, medical_context: Dict, medication_name: str) -> Dict:
        """Get detailed medication information"""
        med_info_prompt = f"""
        Provide detailed information about this medication: {medication_name}
        
        Include:
        - Generic and brand names
        - Common dosages
        - Indications
        - Side effects
        - Contraindications
        
        Format as JSON.
        """
        
        try:
            response = self._call_model(med_info_prompt)
            return json.loads(response.strip())
        except Exception as e:
            return {"error": str(e)}

    def _calculate_dosage(self, medical_context: Dict, parameters: Dict) -> Dict:
        """Calculate dosage recommendations"""
        medication = parameters.get("medication", "")
        weight = parameters.get("weight", 0)
        age = parameters.get("age", 0)
        
        dosage_prompt = f"""
        Calculate appropriate dosage for:
        Medication: {medication}
        Patient Weight: {weight} kg
        Patient Age: {age} years
        
        Provide dosage recommendations with safety considerations.
        Format as JSON.
        """
        
        try:
            response = self._call_model(dosage_prompt)
            return json.loads(response.strip())
        except Exception as e:
            return {"error": str(e)}

    def _check_drug_interactions(self, medical_context: Dict, medications: List) -> Dict:
        """Check for drug interactions"""
        interaction_prompt = f"""
        Check for drug interactions between these medications:
        {json.dumps(medications, indent=2)}
        
        Provide interaction warnings and severity levels.
        Format as JSON.
        """
        
        try:
            response = self._call_model(interaction_prompt)
            return json.loads(response.strip())
        except Exception as e:
            return {"error": str(e)}

    def _get_document_summary(self, medical_context: Dict, doc_id: str) -> Dict:
        """Get summary of specific document"""
        if doc_id in medical_context:
            return {
                "document_id": doc_id,
                "summary": medical_context[doc_id]
            }
        else:
            return {"error": f"Document {doc_id} not found"}


    def get_session_stats(self, session_id: str) -> Dict:
        """Get statistics for a session"""
        if session_id not in self.sessions:
            return {"error": "Session not found"}
        
        session = self.sessions[session_id]
        return {
            "total_questions": len(session['conversation_history']),
            "avg_confidence": sum(q.get('confidence_score', 0) for q in session['conversation_history']) / max(len(session['conversation_history']), 1),
            "created_at": session['created_at'],
            "documents_count": len(session.get('medical_context', {}))
        }
    
    def cleanup_session(self, session_id: str):
        """Clean up session data"""
        if session_id in self.sessions:
            del self.sessions[session_id]
    
    def close(self):
        """Close HTTP client"""
        if self.client:
            self.client.close()

