# AI Medical History Analyzer 🩺📊

An AI-powered tool designed to assist doctors in analyzing patient history by extracting data from medical documents using OCR techniques. Users can ask questions like "What is the patient’s medication history?" or "What is the blood level trend?" to retrieve relevant information from scanned documents.

This project is built with **Python** and **Flask**, and is deployed on **AWS** for real-time usage in healthcare scenarios.

---

## 🚀 Features

✔ Extracts structured information from unorganized medical documents using OCR  
✔ Allows doctors and users to interactively ask questions about patient history  
✔ Provides insights based on extracted data such as medication records, test results, and more  
✔ Lightweight and efficient – optimized for deployment on cloud services

---

## 🛠 Tech Stack

- **Programming Language:** Python  
- **Web Framework:** Flask  
- **OCR:** Various Python OCR libraries (as listed in `req.txt`)  
- **Deployment:** AWS services  
- **Database:** Configurable as needed (local or cloud-hosted)

---

## 📂 Project Structure

Med-Analyser
- templates         # HTML templates for UI 
- .gitignore
- agent.py            # Helper module for agent logic
- file_processing.py  # File handling and OCR processing
- req.txt            # Required Python libraries
- server.py          # Main application server file
- README.md


---

## ✅ Prerequisites

- Python 3.8 or higher  
- `pip` package manager  
- Access to AWS (for deployment; optional for local use)  
- `.env` file with model configurations and API keys

---

## 📥 Installation

1. **Clone the repository**

   ```bash
   git clone https://github.com/ray-27/Med-Analyser.git
   cd Med-Analyser
   ```
2. Create a virtual environment (optional but recommended)
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. Install the dependencies
  ```bash
  pip install -r req.txt
  ```
4. Create a .env file in the project root with the following content:
  ```bash
    MODEL_SERVICE=gemini
  # options: ollama, gemini, other
  
  GEMINI_LLM_MODEL=gemini-2.0-flash
  GEMINI_AGENT_MODEL=gemini-2.0-flash-exp
  GEMINI_API_KEY=
  
  OLLAMA_ENDPOINT=http://localhost:11434
  OLLAMA_MODEL=hf.co/unsloth/medgemma-4b-it-GGUF:Q4_K_M
  ```

---

▶ How to Run Locally

Ensure the .env file is configured with appropriate model settings.

Start the server by running:

```bash
python server.py
```
Open your browser and navigate to:

`http://127.0.0.1:5000`

Upload patient documents and interact with the analyzer using natural language queries.

---

📦 Deployment

The app is ready to be deployed on cloud platforms like AWS EC2, Elastic Beanstalk, or similar. Ensure environment variables are securely configured, and OCR services and models are accessible.

🤝 Contributing

Contributions are welcome! Fork this repository, report issues, or submit pull requests to help improve this project.

📫 Contact

Feel free to connect with me on GitHub
 for questions or collaboration.
