import google.generativeai as genai
from PIL import Image
import io
import base64
import fitz  # PyMuPDF
from pdf2image import convert_from_path
import tempfile
import os
from dotenv import load_dotenv

class FileTools:
    def __init__(self):
        """Initialize the OCR tool with Gemini Pro 2.5"""
        load_dotenv()   
        api_key = os.getenv('GEMINI_API_KEY')
        model = os.getenv('GEMINI_LLM_MODEL')

        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model)
    
    def extract_text_from_image(self, image_path: str = None, image_bytes: bytes = None) -> dict:
        """
        Extract text from image using Gemini Pro 2.5 OCR capabilities
        
        Args:
            image_path: Path to image file
            image_bytes: Image as bytes (alternative to file path)
            
        Returns:
            dict: Contains extracted text and confidence info
        """
        try:
            # Load image
            if image_path:
                image = Image.open(image_path)
            elif image_bytes:
                image = Image.open(io.BytesIO(image_bytes))
            else:
                raise ValueError("Either image_path or image_bytes must be provided")
            
            # Prepare prompt for OCR
            prompt = """
            Please extract all text from this image using OCR. 
            Provide the text in a structured format, preserving:
            - Line breaks and spacing where meaningful
            - Any table structures if present
            - Headers and formatting hierarchy
            
            If the image contains handwritten text, do your best to transcribe it.
            If text is unclear or uncertain, indicate this with [unclear] tags.
            """
            
            # Generate response
            response = self.model.generate_content([prompt, image])
            
            return {
                "success": True,
                "extracted_text": response.text,
                "method": "gemini_pro_2.5_ocr",
                "error": None
            }
            
        except Exception as e:
            return {
                "success": False,
                "extracted_text": "",
                "method": "gemini_pro_2.5_ocr",
                "error": str(e)
            }
    
    def extract_with_structure_analysis(self, image_path: str = None, image_bytes: bytes = None) -> dict:
        """
        Extract text with additional structure analysis for complex documents
        """
        try:
            if image_path:
                image = Image.open(image_path)
            elif image_bytes:
                image = Image.open(io.BytesIO(image_bytes))
            else:
                raise ValueError("Either image_path or image_bytes must be provided")
            
            prompt = """
            Analyze this image and extract text with structural information:
            
            1. First, identify the document type (form, invoice, receipt, handwritten note, etc.)
            2. Extract all text content
            3. Identify key-value pairs if present
            4. Preserve table structures with proper formatting
            5. Note any signatures, stamps, or special markings
            
            Format the output as JSON with these fields:
            - document_type: string
            - extracted_text: string (full text)
            - key_value_pairs: object (if applicable)
            - tables: array (if present)
            - special_elements: array (signatures, stamps, etc.)
            """
            
            response = self.model.generate_content([prompt, image])
            
            return {
                "success": True,
                "analysis": response.text,
                "method": "gemini_pro_2.5_structured_ocr",
                "error": None
            }
            
        except Exception as e:
            return {
                "success": False,
                "analysis": "",
                "method": "gemini_pro_2.5_structured_ocr",
                "error": str(e)
            }
    
    def extract_text_from_pdf(self, pdf_path: str, use_ocr_fallback: bool = True) -> dict:
        """
        Extract text from PDF with OCR fallback for scanned documents
        
        Args:
            pdf_path: Path to PDF file
            use_ocr_fallback: Whether to use OCR if traditional extraction fails
            
        Returns:
            dict: Contains extracted text and method used
        """
        try:
            # First, try traditional text extraction
            doc = fitz.open(pdf_path)
            traditional_text = ""
            
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                traditional_text += f"\n--- Page {page_num + 1} ---\n"
                traditional_text += page.get_text()
            
            doc.close()
            
            # Check if traditional extraction was successful
            if traditional_text.strip() and len(traditional_text.strip()) > 50:
                return {
                    "success": True,
                    "extracted_text": traditional_text,
                    "method": "traditional_pdf_extraction",
                    "pages_processed": len(doc),
                    "error": None
                }
            
            # If traditional extraction failed and OCR fallback is enabled
            elif use_ocr_fallback:
                return self._extract_with_ocr_fallback(pdf_path)
            
            else:
                return {
                    "success": False,
                    "extracted_text": traditional_text,
                    "method": "traditional_pdf_extraction",
                    "error": "Insufficient text extracted and OCR fallback disabled"
                }
                
        except Exception as e:
            if use_ocr_fallback:
                return self._extract_with_ocr_fallback(pdf_path)
            else:
                return {
                    "success": False,
                    "extracted_text": "",
                    "method": "traditional_pdf_extraction",
                    "error": str(e)
                }
    
    def _extract_with_ocr_fallback(self, pdf_path: str) -> dict:
        """Use OCR fallback method for scanned PDFs"""
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                images = convert_from_path(pdf_path, output_folder=temp_dir)
                
                all_text = ""
                processed_pages = 0
                
                for i, image in enumerate(images):
                    temp_image_path = os.path.join(temp_dir, f"page_{i+1}.png")
                    image.save(temp_image_path, "PNG")
                    
                    # Fix: Call method directly on self
                    ocr_result = self.extract_text_from_image(temp_image_path)
                    
                    if ocr_result["success"]:
                        all_text += f"\n--- Page {i + 1} (OCR) ---\n"
                        all_text += ocr_result["extracted_text"]
                        processed_pages += 1
                    
                return {
                    "success": processed_pages > 0,
                    "extracted_text": all_text,
                    "method": "gemini_2.0_flash_pdf_ocr",
                    "pages_processed": processed_pages,
                    "total_pages": len(images),
                    "error": None if processed_pages > 0 else "Failed to process any pages"
                }
                
        except Exception as e:
            return {
                "success": False,
                "extracted_text": "",
                "method": "gemini_2.0_flash_pdf_ocr",
                "error": str(e)
            }
    
    def extract_structured_pdf(self, pdf_path: str) -> dict:
        """Extract structured information from PDF using OCR analysis"""
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                images = convert_from_path(pdf_path, output_folder=temp_dir)
                
                structured_data = {
                    "pages": [],
                    "document_analysis": None,
                    "method": "gemini_2.0_flash_structured_pdf_ocr"
                }
                
                for i, image in enumerate(images):
                    temp_image_path = os.path.join(temp_dir, f"page_{i+1}.png")
                    image.save(temp_image_path, "PNG")
                    
                    # Fix: Call method directly on self
                    analysis = self.extract_with_structure_analysis(temp_image_path)
                    
                    if analysis["success"]:
                        structured_data["pages"].append({
                            "page_number": i + 1,
                            "analysis": analysis["analysis"]
                        })
                
                return {
                    "success": len(structured_data["pages"]) > 0,
                    "structured_data": structured_data,
                    "error": None
                }
                
        except Exception as e:
            return {
                "success": False,
                "structured_data": None,
                "error": str(e)
            }

    import json

    def extract_medical_prescription_data(self, image_path: str = None, image_bytes: bytes = None) -> dict:
        """
        Extract patient description and prescribed medications from medical images/reports
        
        Args:
            image_path: Path to medical image file
            image_bytes: Image as bytes (alternative to file path)
            
        Returns:
            dict: Contains patient info and prescriptions
        """
        try:
            # Load image
            if image_path:
                image = Image.open(image_path)
            elif image_bytes:
                image = Image.open(io.BytesIO(image_bytes))
            else:
                raise ValueError("Either image_path or image_bytes must be provided")
            
            # Specialized prompt for medical data extraction
            prompt = """
            Analyze this medical document/prescription and extract the following information:
            
            **PATIENT INFORMATION:**
            - Patient name
            - Age/Date of birth
            - Gender
            - Patient ID (if present)
            - Address/Contact (if present)
            
            **MEDICAL DESCRIPTION/DIAGNOSIS:**
            - Chief complaint
            - Symptoms described
            - Diagnosis
            - Medical history mentioned
            - Vital signs (if present)
            
            **PRESCRIBED MEDICATIONS:**
            For each medication, extract:
            - Medication name
            - Dosage/Strength
            - Frequency (how often to take)
            - Duration (how long to take)
            - Special instructions
            
            **ADDITIONAL INSTRUCTIONS:**
            - Doctor's advice
            - Follow-up instructions
            - Lifestyle recommendations
            
            Format the response as structured JSON with these exact keys:
            {
                "patient_info": {
                    "name": "",
                    "age": "",
                    "gender": "",
                    "patient_id": "",
                    "contact": ""
                },
                "medical_description": {
                    "chief_complaint": "",
                    "symptoms": [],
                    "diagnosis": "",
                    "medical_history": "",
                    "vital_signs": {}
                },
                "prescriptions": [
                    {
                        "medication": "",
                        "dosage": "",
                        "frequency": "",
                        "duration": "",
                        "instructions": ""
                    }
                ],
                "doctor_instructions": "",
                "follow_up": "",
                "doctor_name": "",
                "date": ""
            }
            
            If any information is unclear or not present, use "Not specified" or leave empty array/object.
            """
            
            # Generate response
            response = self.model.generate_content([prompt, image])
            
            return {
                "success": True,
                "medical_data": response.text,
                "method": "gemini_medical_extraction",
                "error": None
            }
            
        except Exception as e:
            return {
                "success": False,
                "medical_data": None,
                "method": "gemini_medical_extraction",
                "error": str(e)
            }

    def extract_prescription_summary(self, image_path: str = None, image_bytes: bytes = None) -> dict:
        """
        Get a quick summary of patient condition and medications only
        """

        print("the summarization reached")
        try:
            # Load image
            if image_path:
                image = Image.open(image_path)
            elif image_bytes:
                image = Image.open(io.BytesIO(image_bytes))
            else:
                raise ValueError("Either image_path or image_bytes must be provided")
            
            # Simplified prompt for quick extraction
            prompt = """
            From this medical document, provide a concise summary with only:
            
            Take out patient related details like name, age, sex, and more in tabular form
            1. **Patient Condition/Problem:**
            What is the patient's main health issue or complaint?
            
            2. **Prescribed Medications:**
            List all medications with dosage and frequency in simple format:
            - Medicine Name (Dosage) - How often to take
            - When the prescription was given (date of writing the prescription)

            3. **Medical report:**
            - List out any metrics like blood pressure, blood count, sugar level or anything similar
            - Store these in a tabular form
            
            Make response as descriptive as possible, if there are any table then have that stored in tabular form in the text itself.

            IF THE IMAGE IS OF ANY TECHINICAL REPORT LIKE X-RAY or ECG THEN DONT ANALYSE IT JUST WRITE THE RESPECTIVE TYPE OF FILE (either x-ray ECG etc. )
            """
            
            print("doing analysis")
            response = self.model.generate_content([prompt, image])
            
            return {
                "success": True,
                "summary": str(response.text).strip(),
                "method": "gemini_prescription_summary",
                "error": None
            }
            
        except Exception as e:
            return {
                "success": False,
                "summary": None,
                "method": "gemini_prescription_summary",
                "error": str(e)
            }
    


    def extract_medical_pdf_data(self, pdf_path: str) -> dict:
        """
        Extract medical data from PDF reports with OCR fallback
        """
        try:
            # First try traditional PDF text extraction
            doc = fitz.open(pdf_path)
            traditional_text = ""
            
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                traditional_text += page.get_text()
            
            doc.close()
            
            # If we got good text extraction, analyze it directly
            if traditional_text.strip() and len(traditional_text.strip()) > 50:
                return self._analyze_medical_text(traditional_text)
            
            # Otherwise use OCR fallback
            else:
                with tempfile.TemporaryDirectory() as temp_dir:
                    images = convert_from_path(pdf_path, output_folder=temp_dir)
                    
                    all_medical_data = []
                    
                    for i, image in enumerate(images):
                        temp_image_path = os.path.join(temp_dir, f"page_{i+1}.png")
                        image.save(temp_image_path, "PNG")
                        
                        medical_result = self.extract_medical_prescription_data(temp_image_path)
                        
                        if medical_result["success"]:
                            all_medical_data.append({
                                "page": i + 1,
                                "data": medical_result["medical_data"]
                            })
                    
                    return {
                        "success": len(all_medical_data) > 0,
                        "medical_data": all_medical_data,
                        "method": "pdf_medical_ocr_extraction",
                        "error": None
                    }
                    
        except Exception as e:
            return {
                "success": False,
                "medical_data": None,
                "method": "pdf_medical_extraction",
                "error": str(e)
            }

    def _analyze_medical_text(self, text: str) -> dict:
        """
        Analyze extracted text for medical information
        """
        try:
            prompt = f"""
            Analyze this medical text and extract patient description and prescriptions:
            
            {text}
            
            Extract:
            1. Patient condition/diagnosis
            2. All prescribed medications with dosages
            
            Format as JSON with patient_condition and prescriptions fields.
            """
            
            response = self.model.generate_content(prompt)
            
            return {
                "success": True,
                "medical_data": response.text,
                "method": "text_medical_analysis",
                "error": None
            }
            
        except Exception as e:
            return {
                "success": False,
                "medical_data": None,
                "method": "text_medical_analysis",
                "error": str(e)
            }
    
    def extract_pdf_prescription_summary(self, pdf_path: str) -> dict:
        """
        Extract a quick summary of patient condition and medications from PDF files
        
        Args:
            pdf_path: Path to PDF medical document
            
        Returns:
            dict: Contains patient condition and prescription summary
        """
        try:
            # First, try traditional text extraction
            doc = fitz.open(pdf_path)
            traditional_text = ""
            
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                traditional_text += f"\n--- Page {page_num + 1} ---\n"
                traditional_text += page.get_text()
            
            doc.close()
            
            # Check if traditional extraction was successful
            if traditional_text.strip() and len(traditional_text.strip()) > 50:
                return self._analyze_medical_text_summary(traditional_text, "traditional_pdf_extraction")
            
            # Use OCR fallback if traditional extraction failed
            else:
                return self._extract_pdf_summary_with_ocr(pdf_path)
                
        except Exception as e:
            # Try OCR fallback on any error
            try:
                return self._extract_pdf_summary_with_ocr(pdf_path)
            except Exception as ocr_error:
                return {
                    "success": False,
                    "summary": None,
                    "method": "pdf_summary_extraction_failed",
                    "error": f"Traditional extraction error: {str(e)}, OCR error: {str(ocr_error)}"
                }

    def _extract_pdf_summary_with_ocr(self, pdf_path: str) -> dict:
        """
        Use OCR to extract summary from PDF when traditional extraction fails
        """
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                # Convert PDF pages to images
                images = convert_from_path(pdf_path, output_folder=temp_dir)
                
                all_summaries = []
                processed_pages = 0
                
                for i, image in enumerate(images):
                    # Save image temporarily
                    temp_image_path = os.path.join(temp_dir, f"page_{i+1}.png")
                    image.save(temp_image_path, "PNG")
                    
                    # Extract summary using existing image OCR method
                    summary_result = self.extract_prescription_summary(temp_image_path)
                    
                    if summary_result["success"]:
                        all_summaries.append({
                            "page": i + 1,
                            "summary": summary_result["summary"]
                        })
                        processed_pages += 1
                
                if processed_pages > 0:
                    # Combine all page summaries
                    combined_summary = self._combine_page_summaries(all_summaries)
                    
                    return {
                        "success": True,
                        "summary": combined_summary,
                        "method": "pdf_ocr_summary_extraction",
                        "pages_processed": processed_pages,
                        "total_pages": len(images),
                        "error": None
                    }
                else:
                    return {
                        "success": False,
                        "summary": None,
                        "method": "pdf_ocr_summary_extraction",
                        "error": "Failed to extract summary from any page"
                    }
                    
        except Exception as e:
            return {
                "success": False,
                "summary": None,
                "method": "pdf_ocr_summary_extraction",
                "error": str(e)
            }

    def _analyze_medical_text_summary(self, text: str, method: str) -> dict:
        """
        Analyze extracted PDF text for patient condition and prescriptions summary
        """
        try:
            prompt = f"""
            From this medical document text, provide a concise summary with only:
            
            1. **Patient Condition/Problem:**
            What is the patient's main health issue, complaint, or diagnosis?
            
            2. **Prescribed Medications:**
            List all medications with dosage and frequency in simple format:
            - Medicine Name (Dosage) - How often to take
            
            Text to analyze:
            {text}
            
            Keep the response brief and focused only on these two aspects.
            If multiple pages contain different information, consolidate into a single coherent summary.
            """
            
            response = self.model.generate_content(prompt)
            
            return {
                "success": True,
                "summary": response.text,
                "method": f"{method}_with_text_analysis",
                "error": None
            }
            
        except Exception as e:
            return {
                "success": False,
                "summary": None,
                "method": f"{method}_with_text_analysis",
                "error": str(e)
            }

    def _combine_page_summaries(self, page_summaries: list) -> str:
        """
        Combine multiple page summaries into a coherent single summary
        """
        try:
            # Prepare all page content for analysis
            all_content = ""
            for page_data in page_summaries:
                all_content += f"\n--- Page {page_data['page']} ---\n{page_data['summary']}"
            
            prompt = f"""
            I have extracted summaries from multiple pages of a medical document. 
            Please combine them into a single, coherent summary focusing on:
            
            1. **Patient Condition/Problem:** (consolidate all diagnoses/complaints)
            2. **Prescribed Medications:** (combine all medications, remove duplicates)
            
            Page summaries:
            {all_content}
            
            Provide a unified summary without mentioning page numbers.
            """
            
            response = self.model.generate_content(prompt)
            return response.text
            
        except Exception as e:
            # Fallback: just concatenate the summaries
            combined = "COMBINED SUMMARY FROM MULTIPLE PAGES:\n\n"
            for page_data in page_summaries:
                combined += f"Page {page_data['page']}:\n{page_data['summary']}\n\n"
            return combined





    
if __name__ == "__main__":
    tools = FileTools()

    # 1. Extract text from image
    # image_result = tools.extract_text_from_image("medimg2.jpg")
    # if image_result["success"]:
    #     print("Extracted text:")
    #     print(image_result["extracted_text"])
    # else:
    #     print("Error:", image_result["error"])

    # # 2. Extract structured data from image (forms, invoices, etc.)
    # structured_result = tools.extract_with_structure_analysis("medimg2.jpg")
    # if structured_result["success"]:
    #     print("Structured analysis:")
    #     print(structured_result["analysis"])

    summary = tools.extract_prescription_summary("medimg2.jpg")
    if summary["success"]:
        print("Patient Condition & Prescriptions:")
        print(summary["summary"])