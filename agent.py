import httpx
from typing import Dict, Any, List, Optional, TypedDict
from langgraph.graph import StateGraph, END, START
import json
import google.generativeai as genai
from datetime import datetime
import os
from dotenv import load_dotenv


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
        print(f"Initialized with {self.api_type} - Model: {self.model_name}")
    
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
            print(f"Processing question: {question} for session: {session_id}")
            
            # Get medical context and document sources
            medical_context = session_data.get('medical_context', {})
            document_sources = session_data.get('document_sources', {})
            conversation_history = session_data.get('conversation_history', [])
            
            print(f"Medical context available: {len(medical_context)} documents")
            
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
            print(f"Error processing question: {e}")
            return {
                "answer": f"I apologize, but I encountered an error: {str(e)}",
                "document_references": [],
                "confidence_score": 0.0,
                "sources_used": 0
            }


    def initialize_session(self, session_id: str, processed_files: List[Dict]):
        """Initialize session with processed medical documents"""
        print(f"Initializing session {session_id} with {len(processed_files)} files")
        
        medical_context = {}
        document_sources = {}
        
        for i, file_data in enumerate(processed_files):
            doc_id = f"doc_{session_id}_{i + 1}"
            
            summary = file_data.get("summary", "")
            print(f"Document {doc_id}: {len(summary)} characters")
            
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
        
        print(f"Session initialized with medical_context keys: {list(medical_context.keys())}")

    def _extract_referenced_documents(self, response: str, document_sources: Dict) -> List[str]:
        """Extract document IDs that are actually referenced in the response"""
        referenced_docs = []
        
        # Look for filename patterns in the response
        for doc_id, source_info in document_sources.items():
            original_name = source_info.get("original_name", "")
            
            # Check if the original filename is mentioned in the response
            if original_name and original_name.lower() in response.lower():
                referenced_docs.append(doc_id)
                print(f"Found reference to document: {original_name} (ID: {doc_id})")
            
            # Also check for document ID patterns (as fallback)
            if doc_id in response:
                referenced_docs.append(doc_id)
                print(f"Found reference to document ID: {doc_id}")
        
        print(f"Total referenced documents: {len(referenced_docs)}")
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


# import httpx
# from typing import Dict, Any, List, Optional, TypedDict
# from langgraph.graph import StateGraph, END, START
# import json
# import google.generativeai as genai
# from datetime import datetime
# import asyncio
# import os
# from dotenv import load_dotenv

# class MedicalAgentState(TypedDict):
#     """Enhanced state for medical agent with memory and document tracking"""
#     session_id: str
#     current_question: str
#     conversation_history: List[Dict]
#     medical_context: Dict[str, Any]  # Document ID -> content mapping
#     document_sources: Dict[str, Any]  # Document metadata
#     extracted_context: List[Dict]  # Context with source references
#     response: str
#     document_references: List[str]  # Referenced document IDs
#     confidence_score: float
#     needs_clarification: bool
#     error_message: str
#     tool_calls: List[Dict]
#     tool_results: List[Dict]
#     question_analysis: Dict[str, Any]

# class MedicalAgent:
#     def __init__(self, 
#                  model_endpoint: str = None,
#                  model_name: str = None,
#                  api_type: str = None):
#         """Initialize agent with flexible model support"""
#         load_dotenv()
        
#         # Auto-detect API type based on environment or parameters
#         self.gemini_api_key = os.getenv('GEMINI_API_KEY')
#         self.ollama_endpoint = os.getenv('OLLAMA_ENDPOINT', 'http://localhost:11434')
#         self.vllm_endpoint = os.getenv('VLLM_ENDPOINT', 'http://localhost:8000/v1')
        
#         # Determine which API to use
#         if api_type == 'ollama' or (not self.gemini_api_key and not model_endpoint):
#             self.api_type = 'ollama'
#             self.endpoint = self.ollama_endpoint
#             self.model_name = model_name or os.getenv('OLLAMA_MODEL', 'llama3-groq-tool-use:8b')
#         elif api_type == 'gemini' or self.gemini_api_key:
#             self.api_type = 'gemini'
#             self.setup_gemini()
#         else:
#             self.api_type = 'vllm'
#             self.endpoint = model_endpoint or self.vllm_endpoint
#             self.model_name = model_name or os.getenv('VLLM_MODEL', 'google/medgemma-4b-it')
        
#         self.client = httpx.AsyncClient(timeout=30.0)

#         self.sessions = {}  # <-- Add this line to fix the error
        
#         # Initialize workflow and tools
#         self.workflow = self._build_workflow()
#         self.tools = self._initialize_tools()
        
#         # Medical knowledge base for validation
#         self.medical_keywords = [
#             "medication", "dosage", "prescription", "diagnosis", "symptoms",
#             "treatment", "condition", "patient", "doctor", "follow-up"
#         ]
        
#         # Document storage with source tracking
#         self.document_storage = {}
#         print(f"Initialized with {self.api_type} - Model: {self.model_name}")
    
#     async def _call_model(self, prompt: str, max_tokens: int = 5120, temperature: float = 0.7) -> str:
#         """Call the appropriate model API"""
#         if self.api_type == 'ollama':
#             return await self._call_ollama_model(prompt, max_tokens, temperature)
#         elif self.api_type == 'gemini':
#             return await self._call_gemini_model(prompt, max_tokens, temperature)
#         else:
#             return await self._call_vllm_model(prompt, max_tokens, temperature)
    
#     async def _call_ollama_model(self, prompt: str, max_tokens: int = 512, temperature: float = 0.7) -> str:
#         """Call Ollama API"""
#         payload = {
#             "model": self.model_name,
#             "messages": [{"role": "user", "content": prompt}],
#             "stream": False,
#             "options": {
#                 "num_predict": max_tokens,
#                 "temperature": temperature
#             }
#         }
        
#         try:
#             response = await self.client.post(
#                 f"{self.endpoint}/api/chat",
#                 json=payload
#             )
#             response.raise_for_status()
            
#             result = response.json()
#             return result["message"]["content"]
            
#         except httpx.HTTPError as e:
#             raise Exception(f"Ollama API error: {str(e)}")

#     async def _call_gemini_model(self, prompt: str, max_tokens: int = 512, temperature: float = 0.7) -> str:
#         """Call the Gemini API model"""
#         try:
#             # Configure generation parameters
#             generation_config = genai.types.GenerationConfig(
#                 max_output_tokens=max_tokens,
#                 temperature=temperature
#             )
            
#             response = await asyncio.to_thread(
#                 self.gemini_model.generate_content,
#                 prompt,
#                 generation_config=generation_config
#             )
            
#             return response.text
            
#         except Exception as e:
#             raise Exception(f"Gemini API error: {str(e)}")

#     async def _call_vllm_model(self, prompt: str, max_tokens: int = 512, temperature: float = 0.7) -> str:
#         """Call the vLLM-hosted model"""
#         payload = {
#             "model": self.model_name,
#             "messages": [{"role": "user", "content": prompt}],
#             "max_tokens": max_tokens,
#             "temperature": temperature,
#             "stream": False
#         }
        
#         try:
#             response = await self.client.post(
#                 f"{self.vllm_endpoint}/chat/completions",
#                 json=payload
#             )
#             response.raise_for_status()
            
#             result = response.json()
#             return result["choices"][0]["message"]["content"]
            
#         except httpx.HTTPError as e:
#             raise Exception(f"vLLM API error: {str(e)}")

    
#     def _initialize_tools(self):
#         """Initialize available tools for the agent"""
#         return {
#             "search_documents": self._search_documents,
#             "get_medication_info": self._get_medication_info,
#             "calculate_dosage": self._calculate_dosage,
#             "check_drug_interactions": self._check_drug_interactions,
#             "get_document_summary": self._get_document_summary
#         }
    
#     def _build_workflow(self) -> StateGraph:
#         """Build enhanced LangGraph workflow with memory and tools"""
#         workflow = StateGraph(MedicalAgentState)
        
#         # Define nodes
#         workflow.add_node("load_memory", self._load_conversation_memory)
#         workflow.add_node("analyze_question", self._analyze_question_with_memory)
#         workflow.add_node("determine_tools", self._determine_required_tools)
#         workflow.add_node("execute_tools", self._execute_tools)
#         workflow.add_node("extract_context", self._extract_relevant_context_with_sources)
#         workflow.add_node("generate_response", self._generate_response_with_references)
#         workflow.add_node("validate_response", self._validate_medical_response)
#         workflow.add_node("update_memory", self._update_conversation_memory)
#         workflow.add_node("handle_clarification", self._handle_clarification)
        
#         # Define edges
#         workflow.add_edge(START, "load_memory")
#         workflow.add_edge("load_memory", "analyze_question")
#         workflow.add_edge("analyze_question", "determine_tools")
        
#         # Conditional edge for tool usage
#         workflow.add_conditional_edges(
#             "determine_tools",
#             self._should_use_tools,
#             {
#                 "use_tools": "execute_tools",
#                 "skip_tools": "extract_context"
#             }
#         )
        
#         workflow.add_edge("execute_tools", "extract_context")
#         workflow.add_edge("extract_context", "generate_response")
#         workflow.add_edge("generate_response", "validate_response")
        
#         # Conditional edges for validation
#         workflow.add_conditional_edges(
#             "validate_response",
#             self._should_clarify,
#             {
#                 "clarify": "handle_clarification",
#                 "update_memory": "update_memory"
#             }
#         )
        
#         workflow.add_edge("handle_clarification", "update_memory")
#         workflow.add_edge("update_memory", END)
        
#         return workflow.compile()
    
#     # Memory Management Methods
#     async def _load_conversation_memory(self, state: MedicalAgentState) -> MedicalAgentState:
#         """Load conversation history for the session"""
#         session_id = state["session_id"]
#         print(f"Loading memory for session: {session_id}")  # Debug log


#         if session_id in self.sessions:
#             session_data = self.sessions[session_id]
#             state["conversation_history"] = session_data.get("conversation_history", [])
#             state["medical_context"] = session_data.get("medical_context", {})
#             state["document_sources"] = session_data.get("document_sources", {})
            
#             print(f"Medical context loaded: {len(state['medical_context'])} documents")  # Debug log
#             print(f"Document sources: {state['document_sources'].keys()}")  # Debug log
#         else:
#             print(f"Session {session_id} not found in self.sessions")  # Debug log
#             print(f"Available sessions: {list(self.sessions.keys())}")  # Debug log
        
#             state["conversation_history"] = []
#             state["medical_context"] = {}
#             state["document_sources"] = {}
        
#         return state
    
#     async def _analyze_question_with_memory(self, state: MedicalAgentState) -> MedicalAgentState:
#         """Analyze question considering conversation history"""
#         current_question = state["current_question"]
#         conversation_history = state["conversation_history"]
        
#         # Format conversation history for context
#         history_context = ""
#         if conversation_history:
#             recent_history = conversation_history[-3:]  # Last 3 exchanges
#             for entry in recent_history:
#                 history_context += f"User: {entry['question']}\nAssistant: {entry['answer']}\n\n"
        
#         analysis_prompt = f"""
#         You are a medical AI assistant helping doctors analyze patient medical documents and history.
    
#         Analyze this medical question considering the conversation history:
        
#         Conversation History:
#         {history_context}
        
#         Current Question: {current_question}
        
#         Identify:
#         1. Question type and intent
#         2. References to previous conversation
#         3. Required information sources
#         4. Medical entities mentioned
        
#         Respond in JSON format:
#         {{
#             "question_type": "prescription_inquiry",
#             "references_previous": false,
#             "context_needed": ["prescriptions", "medications"],
#             "medical_entities": ["prescriptions", "medications"],
#             "requires_tools": true
#         }}
#         """
        
#         try:
#             response = await self._call_model(analysis_prompt, max_tokens=300, temperature=0.3)
#             print(f"Raw LLM response: {response}")  # Debug log
            
#             # Clean the response to extract JSON
#             cleaned_response = response.strip()
            
#             # Try to find JSON in the response
#             import re
#             json_match = re.search(r'\{.*\}', cleaned_response, re.DOTALL)
#             if json_match:
#                 json_str = json_match.group()
#                 analysis = json.loads(json_str)
#             else:
#                 # Fallback if no JSON found
#                 print("No JSON found in response, using fallback")
#                 analysis = {
#                     "question_type": "general_medical_inquiry",
#                     "references_previous": False,
#                     "context_needed": ["medical_documents"],
#                     "medical_entities": ["patient", "medical"],
#                     "requires_tools": True
#                 }
            
#             state["question_analysis"] = analysis
#             return state
            
#         except json.JSONDecodeError as json_error:
#             print(f"JSON parsing error: {json_error}")
#             print(f"Raw response: {response}")
            
#             # Fallback analysis if JSON parsing fails
#             state["question_analysis"] = {
#                 "question_type": "general_medical_inquiry",
#                 "references_previous": False,
#                 "context_needed": ["medical_documents"],
#                 "medical_entities": ["patient"],
#                 "requires_tools": True
#             }
#             return state
            
#         except Exception as e:
#             state["error_message"] = f"Question analysis failed: {str(e)}"
#             return state
    
#     # Tool Management Methods
#     async def _determine_required_tools(self, state: MedicalAgentState) -> MedicalAgentState:
#         """Determine which tools are needed based on question analysis"""
#         question_analysis = state.get("question_analysis", {})
#         current_question = state["current_question"]
        
#         tool_selection_prompt = f"""
#         Based on this question analysis, determine which tools to use:
        
#         Question: {current_question}
#         Analysis: {json.dumps(question_analysis, indent=2)}
        
#         Available tools:
#         - search_documents: Search through medical documents
#         - get_medication_info: Get detailed medication information
#         - calculate_dosage: Calculate dosage recommendations
#         - check_drug_interactions: Check for drug interactions
#         - get_document_summary: Get summary of specific document
        
#         Respond with JSON array of required tools and their parameters:
#         [
#             {{
#                 "tool": "tool_name",
#                 "parameters": {{"param1": "value1"}}
#             }}
#         ]
#         """
        
#         try:
#             response = await self._call_model(tool_selection_prompt, temperature=0.3)
#             tools_needed = json.loads(response.strip())
#             state["tool_calls"] = tools_needed
#             return state
#         except Exception as e:
#             state["tool_calls"] = []
#             return state
    
#     def _should_use_tools(self, state: MedicalAgentState) -> str:
#         """Determine if tools should be used"""
#         tool_calls = state.get("tool_calls", [])
#         return "use_tools" if tool_calls else "skip_tools"
    
#     async def _execute_tools(self, state: MedicalAgentState) -> MedicalAgentState:
#         """Execute the determined tools"""
#         tool_calls = state.get("tool_calls", [])
#         tool_results = []
        
#         for tool_call in tool_calls:
#             tool_name = tool_call.get("tool")
#             parameters = tool_call.get("parameters", {})
            
#             if tool_name in self.tools:
#                 try:
#                     result = await self.tools[tool_name](state, parameters)
#                     tool_results.append({
#                         "tool": tool_name,
#                         "parameters": parameters,
#                         "result": result,
#                         "success": True
#                     })
#                 except Exception as e:
#                     tool_results.append({
#                         "tool": tool_name,
#                         "parameters": parameters,
#                         "error": str(e),
#                         "success": False
#                     })
        
#         state["tool_results"] = tool_results
#         return state
    
#     # Tool Implementation Methods
#     async def _search_documents(self, state: MedicalAgentState, parameters: Dict) -> Dict:
#         """Search through medical documents"""
#         query = parameters.get("query", "")
#         medical_context = state["medical_context"]
#         document_sources = state["document_sources"]
        
#         search_results = []
        
#         for doc_id, content in medical_context.items():
#             if query.lower() in content.lower():
#                 search_results.append({
#                     "document_id": doc_id,
#                     "content": content,
#                     "source": document_sources.get(doc_id, {}),
#                     "relevance_score": self._calculate_relevance(query, content)
#                 })
        
#         # Sort by relevance
#         search_results.sort(key=lambda x: x["relevance_score"], reverse=True)
        
#         return {
#             "results": search_results[:5],  # Top 5 results
#             "total_found": len(search_results)
#         }
    
#     async def _get_medication_info(self, state: MedicalAgentState, parameters: Dict) -> Dict:
#         """Get detailed medication information"""
#         medication_name = parameters.get("medication", "")
        
#         med_info_prompt = f"""
#         Provide detailed information about this medication: {medication_name}
        
#         Include:
#         - Generic and brand names
#         - Common dosages
#         - Indications
#         - Side effects
#         - Contraindications
        
#         Format as JSON.
#         """
        
#         try:
#             response = await self._call_model(med_info_prompt)
#             return json.loads(response.strip())
#         except Exception as e:
#             return {"error": str(e)}
    
#     async def _calculate_dosage(self, state: MedicalAgentState, parameters: Dict) -> Dict:
#         """Calculate dosage recommendations"""
#         medication = parameters.get("medication", "")
#         weight = parameters.get("weight", 0)
#         age = parameters.get("age", 0)
        
#         dosage_prompt = f"""
#         Calculate appropriate dosage for:
#         Medication: {medication}
#         Patient Weight: {weight} kg
#         Patient Age: {age} years
        
#         Provide dosage recommendations with safety considerations.
#         Format as JSON.
#         """
        
#         try:
#             response = await self._call_model(dosage_prompt)
#             return json.loads(response.strip())
#         except Exception as e:
#             return {"error": str(e)}
    
#     async def _check_drug_interactions(self, state: MedicalAgentState, parameters: Dict) -> Dict:
#         """Check for drug interactions"""
#         medications = parameters.get("medications", [])
        
#         interaction_prompt = f"""
#         Check for drug interactions between these medications:
#         {json.dumps(medications, indent=2)}
        
#         Provide interaction warnings and severity levels.
#         Format as JSON.
#         """
        
#         try:
#             response = await self._call_model(interaction_prompt)
#             return json.loads(response.strip())
#         except Exception as e:
#             return {"error": str(e)}
    
#     async def _get_document_summary(self, state: MedicalAgentState, parameters: Dict) -> Dict:
#         """Get summary of specific document"""
#         doc_id = parameters.get("document_id", "")
#         medical_context = state["medical_context"]
        
#         if doc_id in medical_context:
#             return {
#                 "document_id": doc_id,
#                 "summary": medical_context[doc_id],
#                 "source": state["document_sources"].get(doc_id, {})
#             }
#         else:
#             return {"error": f"Document {doc_id} not found"}
    
#     # Context and Response Generation Methods
#     async def _extract_relevant_context_with_sources(self, state: MedicalAgentState) -> MedicalAgentState:
#         """Extract relevant context while tracking document sources"""
#         current_question = state["current_question"]
#         tool_results = state.get("tool_results", [])
#         medical_context = state["medical_context"]
        
#         # Combine tool results with medical context
#         all_context = []
#         document_references = []
        
#         # Add tool results
#         for tool_result in tool_results:
#             if tool_result["success"] and "results" in tool_result["result"]:
#                 for result in tool_result["result"]["results"]:
#                     all_context.append({
#                         "content": result["content"],
#                         "document_id": result["document_id"],
#                         "source": result["source"],
#                         "relevance": result.get("relevance_score", 0)
#                     })
#                     document_references.append(result["document_id"])
        
#         # If no tool results, search all documents
#         if not all_context:
#             for doc_id, content in medical_context.items():
#                 relevance = self._calculate_relevance(current_question, content)
#                 if relevance > 0.3:  # Relevance threshold
#                     all_context.append({
#                         "content": content,
#                         "document_id": doc_id,
#                         "source": state["document_sources"].get(doc_id, {}),
#                         "relevance": relevance
#                     })
#                     document_references.append(doc_id)
        
#         # Sort by relevance
#         all_context.sort(key=lambda x: x["relevance"], reverse=True)
        
#         state["extracted_context"] = all_context[:3]  # Top 3 most relevant
#         state["document_references"] = list(set(document_references))  # Remove duplicates
        
#         return state
    
#     def _calculate_relevance(self, query: str, content: str) -> float:
#         """Calculate relevance score between query and content"""
#         query_words = set(query.lower().split())
#         content_words = set(content.lower().split())
        
#         # Simple relevance calculation (can be enhanced with TF-IDF or embeddings)
#         intersection = query_words.intersection(content_words)
#         union = query_words.union(content_words)
        
#         return len(intersection) / len(union) if union else 0
    
#     async def _generate_response_with_references(self, state: MedicalAgentState) -> MedicalAgentState:
#         """Generate response with document references"""
#         current_question = state["current_question"]
#         medical_context = state.get("medical_context", {})
#         conversation_history = state.get("conversation_history", [])
        
#         print(f"Generating response with {len(medical_context)} medical documents")  # Debug log
        
#         if not medical_context:
#             state["response"] = "I apologize, but I don't have access to any medical documents for this session. Please upload documents first."
#             return state
        
#         # Format medical context for the prompt
#         context_text = ""
#         for doc_id, content in medical_context.items():
#             context_text += f"\nDocument {doc_id}:\n{content}\n"
        
#         # Format conversation history
#         history_text = ""
#         if conversation_history:
#             recent_history = conversation_history[-2:]
#             for entry in recent_history:
#                 history_text += f"User: {entry['question']}\nAssistant: {entry['answer']}\n\n"
        
#         response_prompt = f"""You are a medical AI assistant analyzing patient medical documents.

#         Medical Documents Available:
#         {context_text}

#         Previous Conversation:
#         {history_text}

#         Current Question: {current_question}

#         Based on the medical documents provided above, please answer the question accurately. Always reference the specific document ID when citing information (e.g., "According to Document doc_session_1...").

#         If the information is not available in the documents, clearly state that.

#         Answer:"""
        
#         try:
#             response = await self._call_model(response_prompt, max_tokens=800, temperature=0.7)
            
#             state["response"] = response.strip()
#             state["document_references"] = list(medical_context.keys())  # Reference all available docs
#             state["confidence_score"] = 0.8
            
#             return state
#         except Exception as e:
#             state["error_message"] = f"Response generation failed: {str(e)}"
#             return state

    
#     async def _validate_medical_response(self, state: MedicalAgentState) -> MedicalAgentState:
#         """Validate the generated response for medical accuracy"""
#         response = state.get("response", "")
#         current_question = state["current_question"]
#         extracted_context = state.get("extracted_context", [])
        
#         validation_prompt = f"""
#         Evaluate this medical response for:
#         1. Accuracy based on provided context
#         2. Completeness of answer
#         3. Appropriate medical disclaimers
#         4. Need for clarification
        
#         Question: {current_question}
#         Context Available: {len(extracted_context) > 0}
#         Response: {response}
        
#         Respond in JSON format:
#         {{
#             "accuracy_score": "number 1-10",
#             "completeness_score": "number 1-10", 
#             "needs_clarification": true/false,
#             "clarification_reason": "string or null",
#             "overall_quality": "number 1-10"
#         }}
#         """
        
#         try:
#             validation = await self._call_model(validation_prompt)
#             validation_result = json.loads(validation.strip())
            
#             state["validation"] = validation_result
#             state["needs_clarification"] = validation_result.get("needs_clarification", False)
#             state["confidence_score"] = validation_result.get("overall_quality", 5) / 10
            
#             return state
#         except Exception as e:
#             state["needs_clarification"] = False
#             return state
    
#     def _should_clarify(self, state: MedicalAgentState) -> str:
#         """Determine if clarification is needed"""
#         return "clarify" if state.get("needs_clarification", False) else "update_memory"
    
#     async def _handle_clarification(self, state: MedicalAgentState) -> MedicalAgentState:
#         """Handle cases where clarification is needed"""
#         validation = state.get("validation", {})
#         clarification_reason = validation.get("clarification_reason", "")
        
#         clarification_response = f"""
#         I need some clarification to provide a more accurate answer. {clarification_reason}
        
#         Based on the available medical documentation, here's what I can tell you:
#         {state.get("response", "")}
        
#         Could you please provide more specific details about what you're looking for?
#         """
        
#         state["response"] = clarification_response
#         return state
    
#     async def _update_conversation_memory(self, state: MedicalAgentState) -> MedicalAgentState:
#         """Update conversation memory with new Q&A"""
#         session_id = state["session_id"]
#         current_question = state["current_question"]
#         response = state["response"]
#         document_references = state.get("document_references", [])
        
#         # Create conversation entry
#         conversation_entry = {
#             "question": current_question,
#             "answer": response,
#             "timestamp": datetime.now().isoformat(),
#             "document_references": document_references,
#             "confidence_score": state.get("confidence_score", 0.5)
#         }
        
#         # Update session memory
#         if session_id not in self.sessions:
#             self.sessions[session_id] = {
#                 "conversation_history": [],
#                 "medical_context": state["medical_context"],
#                 "document_sources": state["document_sources"],
#                 "created_at": datetime.now().isoformat()
#             }
        
#         self.sessions[session_id]["conversation_history"].append(conversation_entry)
        
#         # Keep only last 10 conversations to manage memory
#         if len(self.sessions[session_id]["conversation_history"]) > 10:
#             self.sessions[session_id]["conversation_history"] = \
#                 self.sessions[session_id]["conversation_history"][-10:]
        
#         return state
    
#     # Public Interface Methods
#     def initialize_session(self, session_id: str, processed_files: List[Dict]):
#         """Initialize session with processed medical documents"""
#         print(f"Initializing session {session_id} with {len(processed_files)} files")  # Debug log

#         medical_context = {}
#         document_sources = {}
        
#         for i, file_data in enumerate(processed_files):
#             doc_id = f"doc_{session_id}_{i + 1}"
            
#             summary = file_data.get("summary", "")
#             print(f"Document {doc_id}: {len(summary)} characters")  # Debug log 
            
#             medical_context[doc_id] = summary
#             document_sources[doc_id] = {
#                 "original_name": file_data.get("original_name", ""),
#                 "file_type": file_data.get("file_type", ""),
#                 "processed_at": file_data.get("processed_at", ""),
#                 "method": file_data.get("method", ""),
#                 "file_path": file_data.get("file_path", "")
#             }
        
#         self.sessions[session_id] = {
#             "medical_context": medical_context,
#             "document_sources": document_sources,
#             "conversation_history": [],
#             "created_at": datetime.now().isoformat()
#         }

#         print(f"Session initialized with medical_context keys: {list(medical_context.keys())}")  
    
#     async def process_question(self, session_id: str, question: str) -> Dict[str, Any]:
#         """Process question with memory and source tracking"""
#         if session_id not in self.sessions:
#             raise ValueError("Session not found. Please upload medical documents first.")
        
#         # Prepare initial state
#         initial_state = MedicalAgentState(
#             session_id=session_id,
#             current_question=question,
#             conversation_history=[],
#             medical_context={},
#             document_sources={},
#             extracted_context=[],
#             response="",
#             document_references=[],
#             confidence_score=0.0,
#             needs_clarification=False,
#             error_message="",
#             tool_calls=[],
#             tool_results=[],
#             question_analysis={}
#         )
        
#         try:
#             # Run the enhanced workflow
#             final_state = await self.workflow.ainvoke(initial_state)
            
#             if final_state.get("error_message"):
#                 return {
#                     "answer": f"I apologize, but I encountered an error: {final_state['error_message']}",
#                     "document_references": [],
#                     "confidence_score": 0.0
#                 }
            
#             return {
#                 "answer": final_state.get("response", "I couldn't generate a response."),
#                 "document_references": final_state.get("document_references", []),
#                 "confidence_score": final_state.get("confidence_score", 0.0),
#                 "sources_used": len(final_state.get("document_references", []))
#             }
            
#         except Exception as e:
#             return {
#                 "answer": f"I apologize, but I encountered an error: {str(e)}",
#                 "document_references": [],
#                 "confidence_score": 0.0
#             }
    
#     def get_session_stats(self, session_id: str) -> Dict:
#         """Get statistics for a session"""
#         if session_id not in self.sessions:
#             return {"error": "Session not found"}
        
#         session = self.sessions[session_id]
#         return {
#             "total_questions": len(session['conversation_history']),
#             "avg_confidence": sum(q.get('confidence_score', 0) for q in session['conversation_history']) / max(len(session['conversation_history']), 1),
#             "created_at": session['created_at'],
#             "documents_count": len(session.get('medical_context', {}))
#         }
    
#     def cleanup_session(self, session_id: str):
#         """Clean up session data"""
#         if session_id in self.sessions:
#             del self.sessions[session_id]
    
#     async def close(self):
#         """Close HTTP client"""
#         await self.client.aclose()
