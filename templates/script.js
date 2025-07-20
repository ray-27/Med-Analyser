class MedicalDocumentChat {
    constructor() {
        this.sessionId = null;
        this.uploadedFiles = [];
        this.documents = [];
        this.isProcessing = false;
        
        this.initializeElements();
        this.setupEventListeners();
        this.setupDragAndDrop();
    }

    initializeElements() {
        // Upload elements
        this.uploadSection = document.getElementById('upload-section');
        this.chatSection = document.getElementById('chat-section');
        this.uploadArea = document.getElementById('upload-area');
        this.fileInput = document.getElementById('file-input');
        this.browseBtnInput = document.getElementById('browse-btn');
        this.fileList = document.getElementById('file-list');
        this.selectedFilesList = document.getElementById('selected-files');
        this.uploadBtn = document.getElementById('upload-btn');
        this.uploadProgress = document.getElementById('upload-progress');
        this.progressFill = document.getElementById('progress-fill');
        this.progressText = document.getElementById('progress-text');

        // Chat elements
        this.chatMessages = document.getElementById('chat-messages');
        this.chatInput = document.getElementById('chat-input');
        this.sendBtn = document.getElementById('send-btn');
        this.confidenceIndicator = document.getElementById('confidence-indicator');

        // Sidebar elements
        this.documentsList = document.getElementById('documents-list');
        this.docCount = document.getElementById('doc-count');
        this.addMoreBtn = document.getElementById('add-more-btn');

        // UI elements
        this.sessionStatus = document.getElementById('session-status');
        this.newSessionBtn = document.getElementById('new-session-btn');
        this.loadingOverlay = document.getElementById('loading-overlay');
        this.loadingText = document.getElementById('loading-text');
        this.errorToast = document.getElementById('error-toast');
        this.errorMessage = document.getElementById('error-message');
        this.closeError = document.getElementById('close-error');

        // Modal elements
        this.documentModal = document.getElementById('document-modal');
        this.modalClose = document.getElementById('modal-close');
        this.modalTitle = document.getElementById('modal-title');
        this.modalFilename = document.getElementById('modal-filename');
        this.modalFileName = document.getElementById('modal-file-name');
        this.modalFileType = document.getElementById('modal-file-type');
        this.modalProcessingMethod = document.getElementById('modal-processing-method');
        this.modalProcessedDate = document.getElementById('modal-processed-date');
        this.modalExtractedText = document.getElementById('modal-extracted-text');
        this.contentLength = document.getElementById('content-length');
        this.copyContent = document.getElementById('copy-content');
        this.downloadContent = document.getElementById('download-content');
    }

    setupEventListeners() {
        // File upload
        this.browseBtnInput.addEventListener('click', () => this.fileInput.click());
        this.fileInput.addEventListener('change', (e) => this.handleFileSelect(e.target.files));
        this.uploadBtn.addEventListener('click', () => this.uploadFiles());

        // Chat
        this.sendBtn.addEventListener('click', () => this.sendMessage());
        this.chatInput.addEventListener('keypress', (e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                this.sendMessage();
            }
        });

        // UI
        this.newSessionBtn.addEventListener('click', () => this.startNewSession());
        this.addMoreBtn.addEventListener('click', () => this.showUploadSection());
        this.closeError.addEventListener('click', () => this.hideError());

        // Modal event listeners
        if (this.modalClose) {
            this.modalClose.addEventListener('click', () => this.hideDocumentModal());
        }
        
        if (this.documentModal) {
            this.documentModal.addEventListener('click', (e) => {
                if (e.target === this.documentModal) {
                    this.hideDocumentModal();
                }
            });
        }
        
        if (this.copyContent) {
            this.copyContent.addEventListener('click', () => this.copyDocumentContent());
        }
        
        if (this.downloadContent) {
            this.downloadContent.addEventListener('click', () => this.downloadDocumentContent());
        }
        
        // Close modal with Escape key
        document.addEventListener('keydown', (e) => {
            if (e.key === 'Escape' && this.documentModal && this.documentModal.classList.contains('show')) {
                this.hideDocumentModal();
            }
        });
    }

    setupDragAndDrop() {
        this.uploadArea.addEventListener('dragover', (e) => {
            e.preventDefault();
            this.uploadArea.classList.add('dragover');
        });

        this.uploadArea.addEventListener('dragleave', (e) => {
            e.preventDefault();
            this.uploadArea.classList.remove('dragover');
        });

        this.uploadArea.addEventListener('drop', (e) => {
            e.preventDefault();
            this.uploadArea.classList.remove('dragover');
            this.handleFileSelect(e.dataTransfer.files);
        });

        this.uploadArea.addEventListener('click', (e) => {
        // Prevent triggering if the browse button was clicked
            if (e.target.closest('#browse-btn')) {
                return;
            }
            this.fileInput.click();
        });
    }

    handleFileSelect(files) {
        this.uploadedFiles = Array.from(files);
        this.displaySelectedFiles();
    }

    displaySelectedFiles() {
        if (this.uploadedFiles.length === 0) {
            this.fileList.style.display = 'none';
            return;
        }

        this.selectedFilesList.innerHTML = '';
        this.uploadedFiles.forEach((file, index) => {
            const li = document.createElement('li');
            li.innerHTML = `
                <div class="file-info">
                    <span class="file-icon">${this.getFileIcon(file.name)}</span>
                    <span>${file.name} (${this.formatFileSize(file.size)})</span>
                </div>
                <button class="remove-file" onclick="medicalChat.removeFile(${index})">Remove</button>
            `;
            this.selectedFilesList.appendChild(li);
        });

        this.fileList.style.display = 'block';
    }

    removeFile(index) {
        this.uploadedFiles.splice(index, 1);
        this.displaySelectedFiles();
    }

    getFileIcon(fileName) {
        // Handle both MIME type and file name
        const name = (fileName || '').toLowerCase();
        
        if (name.endsWith('.pdf')) return '📄';
        if (name.match(/\.(jpg|jpeg|png|gif|bmp)$/)) return '🖼️';
        return '📎';
    }

    formatFileSize(bytes) {
        if (bytes === 0) return '0 B';
        const k = 1024;
        const sizes = ['B', 'KB', 'MB', 'GB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
    }

    async uploadFiles() {
        if (this.uploadedFiles.length === 0 || this.isProcessing) return;

        this.isProcessing = true;
        this.showProgress(0, 'Preparing files...');

        const formData = new FormData();
        this.uploadedFiles.forEach(file => {
            formData.append('files', file);
        });

        try {
            this.updateProgress(30, 'Uploading files...');

            const response = await fetch('/upload', {
                method: 'POST',
                body: formData
            });

            if (!response.ok) {
                throw new Error(`Upload failed: ${response.status} ${response.statusText}`);
            }

            const result = await response.json();

            if (result.status === 'success') {
                this.updateProgress(70, 'Processing documents...');
                this.sessionId = result.session_id;
                this.documents = result.processed_files || [];
                
                setTimeout(() => {
                    this.updateProgress(100, 'Complete!');
                    setTimeout(() => {
                        this.showChatInterface(result);
                        this.hideProgress();
                        this.isProcessing = false;
                    }, 500);
                }, 1000);
            } else {
                throw new Error(result.error || 'Upload failed');
            }
        } catch (error) {
            this.hideProgress();
            this.showError(error.message);
            this.isProcessing = false;
        }
    }

    showProgress(progress, text) {
        this.uploadProgress.style.display = 'block';
        this.updateProgress(progress, text);
    }

    updateProgress(progress, text) {
        this.progressFill.style.width = `${progress}%`;
        this.progressText.textContent = text;
    }

    hideProgress() {
        this.uploadProgress.style.display = 'none';
    }

    showChatInterface(uploadResult) {
        this.uploadSection.style.display = 'none';
        this.chatSection.style.display = 'flex';
        
        this.sessionStatus.textContent = `Session: ${this.sessionId.substring(0, 8)}...`;
        
        // Populate documents sidebar
        this.populateDocumentsSidebar(uploadResult);
        
        // Add welcome message with session info
        this.addMessage('bot', 
            `Great! I've processed ${uploadResult.processed_files} document(s). I can now answer questions about your medical information, prescriptions, diagnoses, and more. What would you like to know?`
        );
    }

    populateDocumentsSidebar(uploadResult) {
        this.docCount.textContent = `${uploadResult.processed_files} documents`;
        this.documentsList.innerHTML = '';

        // Get session info to get detailed file information
        this.fetchSessionInfo();
    }

    // Hide scroll button when at bottom
    handleScroll() {
        const scrollBtn = document.getElementById('scroll-to-bottom');
        if (scrollBtn) {
            const isAtBottom = this.chatMessages.scrollTop + this.chatMessages.clientHeight >= this.chatMessages.scrollHeight - 10;
            scrollBtn.style.display = isAtBottom ? 'none' : 'block';
        }
    }

    async fetchSessionInfo() {
        if (!this.sessionId) {
            console.warn('No session ID available');
            return;
        }

        try {
            console.log(`Fetching session info for: ${this.sessionId}`);

            const response = await fetch(`/session/${this.sessionId}`);
            
            if (!response.ok) {
                const errorText = await response.text();
                throw new Error(`HTTP ${response.status}: ${errorText}`);
            }

            const sessionData = await response.json();
            
            if (sessionData.status === 'error') {
                throw new Error(sessionData.error || 'Session data error');
            }

            console.log('Session data received:', sessionData);
            
            // Check if files exist and is array
            if (sessionData.files && Array.isArray(sessionData.files)) {
                console.log(`Found ${sessionData.files.length} files`);
                this.displayDocuments(sessionData.files);
            } else {
                console.warn('No files found in session data');
                this.displayDocuments([]);
            }
            
        } catch (error) {
            console.error('Failed to fetch session information:', error);
            this.showError(`Failed to load session information: ${error.message}`);
        }
    }

    // Updated displayDocuments method with action buttons
    displayDocuments(files) {
        console.log('Displaying documents:', files);
        this.documentsList.innerHTML = '';
        
        if (!files || files.length === 0) {
            this.documentsList.innerHTML = '<div class="no-documents">No documents uploaded</div>';
            this.docCount.textContent = '0 documents';
            return;
        }
        
        this.docCount.textContent = `${files.length} document${files.length > 1 ? 's' : ''}`;
        
        files.forEach((file, index) => {
            const docElement = document.createElement('div');
            docElement.className = 'document-item';
            docElement.dataset.docId = file.file_id;
            
            const fileType = file.original_name.split('.').pop().toUpperCase();
            const processedDate = new Date(file.processed_at).toLocaleString();
            
            docElement.innerHTML = `
                <div class="document-name">
                    ${this.getFileIcon(file.original_name)} ${file.original_name}
                    <span class="document-type">${fileType}</span>
                </div>
                ${file.summary ? `<div class="document-summary">${file.summary.substring(0, 150)}...</div>` : ''}
                <div class="document-meta">
                    <span>Processed: ${processedDate}</span>
                    <span>${file.method || 'OCR'}</span>
                </div>
                <div class="document-actions">
                    <button class="btn-view-modal" title="View in Modal"></button>
                    <button class="btn-open-tab" title="Open in New Tab">👁️ View file in a new tab</button>
                </div>
            `;
            
            // Modal click handler (existing functionality)
            const viewModalBtn = docElement.querySelector('.btn-view-modal');
            viewModalBtn.addEventListener('click', (e) => {
                e.preventDefault();
                e.stopPropagation();
                this.showDocumentModal(file);
            });
            
            // New tab click handler (new functionality)
            const openTabBtn = docElement.querySelector('.btn-open-tab');
            openTabBtn.addEventListener('click', (e) => {
                e.preventDefault();
                e.stopPropagation();
                this.openDocumentInNewTab(file);
            });
            
            this.documentsList.appendChild(docElement);
        });
    }

    // Document viewer methods
    openDocumentInNewTab(fileData) {
        console.log('Opening document in new tab:', fileData);
        
        // Create HTML content for the new tab
        const htmlContent = this.generateDocumentHTML(fileData);
        
        // Open new window/tab
        const newTab = window.open('', '_blank');
        
        if (newTab) {
            newTab.document.write(htmlContent);
            newTab.document.close();
            newTab.focus();
        } else {
            // Fallback if popup blocked
            this.showError('Please allow popups to open documents in new tabs');
            // Fall back to modal
            this.showDocumentModal(fileData);
        }
    }

    generateDocumentHTML(fileData) {
        const fileName = fileData.original_name || 'Unknown Document';
        const fileType = fileName.split('.').pop()?.toUpperCase() || 'UNKNOWN';
        const processedDate = new Date(fileData.processed_at || new Date()).toLocaleString();
        const extractedText = fileData.summary || 'No content extracted';
        const method = fileData.method || 'OCR';
        
        // Generate document viewer URL
        const documentUrl = `/view-document/${this.sessionId}/${this.sessionId}_${fileName}`;
        const isPDF = fileType.toLowerCase() === 'pdf';
        const isImage = ['JPG', 'JPEG', 'PNG', 'GIF', 'BMP'].includes(fileType);
        
        // Create document viewer section
        const documentViewer = this.generateDocumentViewer(documentUrl, isPDF, isImage, fileName);
        
        return `
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Medical Document: ${fileName}</title>
    <style>
        /* Comprehensive CSS for document viewer - same as before */
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background: #f8fafc; color: #2d3748; line-height: 1.6; padding: 20px; }
        .container { max-width: 1400px; margin: 0 auto; background: white; border-radius: 12px; box-shadow: 0 4px 20px rgba(0, 0, 0, 0.1); overflow: hidden; }
        .header { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 30px; text-align: center; }
        .document-title { font-size: 1.8rem; font-weight: 600; margin-bottom: 10px; display: flex; align-items: center; justify-content: center; gap: 15px; }
        .main-content { display: grid; grid-template-columns: 1fr 1fr; gap: 0; min-height: 800px; }
        .document-viewer { background: #f8fafc; border-right: 1px solid #e2e8f0; display: flex; flex-direction: column; }
        .viewer-header { padding: 20px; background: #edf2f7; border-bottom: 1px solid #e2e8f0; font-weight: 600; color: #2d3748; display: flex; justify-content: space-between; align-items: center; }
        .viewer-content { flex: 1; padding: 20px; overflow: auto; display: flex; justify-content: center; align-items: flex-start; }
        .pdf-viewer { width: 100%; height: 100%; min-height: 600px; border: none; border-radius: 8px; }
        .image-viewer { max-width: 100%; height: auto; border-radius: 8px; box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1); cursor: zoom-in; }
        .image-viewer.zoomed { cursor: zoom-out; transform: scale(1.5); transition: transform 0.3s ease; }
        .analysis-panel { display: flex; flex-direction: column; }
        .panel-header { padding: 20px; background: #edf2f7; border-bottom: 1px solid #e2e8f0; font-weight: 600; color: #2d3748; }
        .panel-content { flex: 1; padding: 20px; overflow: auto; }
        .info-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 15px; margin-bottom: 25px; }
        .info-card { background: #f8fafc; padding: 15px; border-radius: 8px; border: 1px solid #e2e8f0; }
        .info-label { font-size: 0.85rem; color: #718096; font-weight: 500; margin-bottom: 5px; text-transform: uppercase; letter-spacing: 0.5px; }
        .info-value { font-size: 0.95rem; color: #2d3748; font-weight: 600; }
        .processing-method { display: inline-flex; align-items: center; gap: 5px; padding: 4px 10px; border-radius: 12px; font-size: 0.8rem; font-weight: 500; background: #feebc8; color: #c05621; }
        .extracted-content { background: #ffffff; border: 1px solid #e2e8f0; border-radius: 8px; padding: 20px; }
        .content-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 15px; padding-bottom: 10px; border-bottom: 2px solid #e2e8f0; }
        .content-title { font-size: 1.1rem; font-weight: 600; color: #2d3748; }
        .content-length { font-size: 0.8rem; color: #718096; background: #edf2f7; padding: 3px 8px; border-radius: 10px; }
        .extracted-text { line-height: 1.7; color: #4a5568; white-space: pre-wrap; font-size: 0.95rem; background: #f8fafc; padding: 15px; border-radius: 6px; border: 1px solid #e2e8f0; max-height: 400px; overflow-y: auto; }
        .actions { display: flex; gap: 15px; justify-content: center; margin-top: 25px; padding-top: 20px; border-top: 1px solid #e2e8f0; }
        .btn { padding: 10px 20px; border: none; border-radius: 8px; cursor: pointer; font-size: 0.9rem; font-weight: 500; transition: all 0.3s ease; text-decoration: none; display: inline-flex; align-items: center; gap: 8px; }
        .btn-copy { background: #667eea; color: white; }
        .btn-copy:hover { background: #5a6fd8; transform: translateY(-2px); }
        .btn-download { background: #48bb78; color: white; }
        .btn-download:hover { background: #38a169; transform: translateY(-2px); }
        .btn-print { background: #ed8936; color: white; }
        .btn-print:hover { background: #dd7324; transform: translateY(-2px); }
        .zoom-controls { display: flex; gap: 10px; }
        .zoom-btn { background: #667eea; color: white; border: none; padding: 5px 10px; border-radius: 4px; cursor: pointer; font-size: 0.8rem; }
        .zoom-btn:hover { background: #5a6fd8; }
        .error-message { background: #fed7d7; color: #c53030; padding: 15px; border-radius: 8px; text-align: center; }
        @media (max-width: 1024px) { .main-content { grid-template-columns: 1fr; grid-template-rows: auto auto; } .document-viewer { border-right: none; border-bottom: 1px solid #e2e8f0; max-height: 400px; } }
        @media print { .actions, .zoom-controls { display: none; } .main-content { grid-template-columns: 1fr; } .document-viewer { display: none; } }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1 class="document-title">
                <span class="file-icon">${this.getFileIcon(fileName)}</span>
                ${fileName}
            </h1>
            <p class="subtitle">Medical Document Analysis with Source Verification</p>
        </div>
        
        <div class="main-content">
            <div class="document-viewer">
                <div class="viewer-header">
                    <span>Original Document</span>
                    ${isImage ? '<div class="zoom-controls"><button class="zoom-btn" onclick="toggleZoom()">🔍 Zoom</button></div>' : ''}
                </div>
                <div class="viewer-content">
                    ${documentViewer}
                </div>
            </div>
            
            <div class="analysis-panel">
                <div class="panel-header">AI Analysis Results</div>
                <div class="panel-content">
                    <div class="info-grid">
                        <div class="info-card">
                            <div class="info-label">File Name</div>
                            <div class="info-value">${fileName}</div>
                        </div>
                        <div class="info-card">
                            <div class="info-label">File Type</div>
                            <div class="info-value">${fileType}</div>
                        </div>
                        <div class="info-card">
                            <div class="info-label">Processing Method</div>
                            <div class="info-value">
                                <span class="processing-method">${method}</span>
                            </div>
                        </div>
                        <div class="info-card">
                            <div class="info-label">Processed Date</div>
                            <div class="info-value">${processedDate}</div>
                        </div>
                    </div>
                    
                    <div class="extracted-content">
                        <div class="content-header">
                            <h2 class="content-title">Extracted Medical Information</h2>
                            <span class="content-length">${extractedText.length} characters</span>
                        </div>
                        <div class="extracted-text">${extractedText}</div>
                    </div>
                    
                    <div class="actions">
                        <button class="btn btn-copy" onclick="copyToClipboard()">📋 Copy Text</button>
                        <button class="btn btn-download" onclick="downloadText()">💾 Download</button>
                        <button class="btn btn-print" onclick="window.print()">🖨️ Print</button>
                    </div>
                </div>
            </div>
        </div>
    </div>
    
    <script>
        let isZoomed = false;
        function toggleZoom() {
            const image = document.querySelector('.image-viewer');
            if (image) {
                isZoomed = !isZoomed;
                image.classList.toggle('zoomed', isZoomed);
            }
        }
        function copyToClipboard() {
            const textContent = \`${extractedText.replace(/`/g, '\\`')}\`;
            navigator.clipboard.writeText(textContent).then(() => {
                alert('Text copied to clipboard!');
            }).catch(err => {
                console.error('Failed to copy text: ', err);
                alert('Failed to copy text');
            });
        }
        function downloadText() {
            const textContent = \`${extractedText.replace(/`/g, '\\`')}\`;
            const blob = new Blob([textContent], { type: 'text/plain' });
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = '${fileName}_extracted.txt';
            document.body.appendChild(a);
            a.click();
            document.body.removeChild(a);
            URL.revokeObjectURL(url);
        }
    </script>
</body>
</html>`;
    }

    generateDocumentViewer(documentUrl, isPDF, isImage, fileName) {
        if (isPDF) {
            return `
                <iframe 
                    class="pdf-viewer" 
                    src="${documentUrl}" 
                    title="PDF Viewer for ${fileName}"
                    onload="console.log('PDF loaded successfully')"
                    onerror="this.style.display='none'; this.nextElementSibling.style.display='block';">
                </iframe>
                <div class="error-message" style="display: none;">
                    <p>Unable to display PDF. <a href="${documentUrl}" target="_blank">Click here to download</a></p>
                </div>
            `;
        } else if (isImage) {
            return `
                <img 
                    class="image-viewer" 
                    src="${documentUrl}" 
                    alt="Medical document: ${fileName}"
                    onclick="toggleZoom()"
                    onload="console.log('Image loaded successfully')"
                    onerror="this.style.display='none'; this.nextElementSibling.style.display='block';">
                <div class="error-message" style="display: none;">
                    <p>Unable to display image. <a href="${documentUrl}" target="_blank">Click here to download</a></p>
                </div>
            `;
        } else {
            return `
                <div class="error-message">
                    <p>Preview not available for this file type.</p>
                    <p><a href="${documentUrl}" target="_blank" class="btn btn-download">📥 Download File</a></p>
                </div>
            `;
        }
    }

    // Modal functionality methods
    showDocumentModal(fileData) {
        console.log('showDocumentModal called with:', fileData);
        
        if (!this.documentModal) {
            console.error('Modal element not found!');
            return;
        }
        
        // Update modal content
        if (this.modalFilename) {
            this.modalFilename.textContent = fileData.original_name || 'Unknown File';
        }
        
        if (this.modalFileName) {
            this.modalFileName.textContent = fileData.original_name || 'Unknown File';
        }
        
        if (this.modalFileType) {
            const fileType = (fileData.original_name || '').split('.').pop()?.toUpperCase() || 'UNKNOWN';
            this.modalFileType.textContent = fileType;
        }
        
        if (this.modalProcessingMethod) {
            const method = fileData.method || 'OCR';
            this.modalProcessingMethod.textContent = method;
            this.modalProcessingMethod.className = `processing-status ${method.toLowerCase().includes('traditional') ? 'success' : 'ocr'}`;
        }
        
        if (this.modalProcessedDate) {
            const processedDate = new Date(fileData.processed_at || new Date()).toLocaleString();
            this.modalProcessedDate.textContent = processedDate;
        }
        
        if (this.modalExtractedText && this.contentLength) {
            const extractedText = fileData.summary || 'No content extracted';
            this.modalExtractedText.textContent = extractedText;
            this.contentLength.textContent = `${extractedText.length} characters`;
        }
        
        // Store current file data
        this.currentModalFile = fileData;
        
        // Show modal with class toggle
        this.documentModal.classList.add('show');
        this.documentModal.style.display = 'flex';
        document.body.style.overflow = 'hidden';
        
        console.log('Modal should now be visible');
    }

    hideDocumentModal() {
        if (this.documentModal) {
            this.documentModal.classList.remove('show');
            this.documentModal.style.display = 'none';
            document.body.style.overflow = 'auto';
        }
        this.currentModalFile = null;
    }

    async copyDocumentContent() {
        if (!this.currentModalFile) return;
        
        const content = this.currentModalFile.summary || '';
        
        try {
            await navigator.clipboard.writeText(content);
            
            const originalText = this.copyContent.textContent;
            this.copyContent.textContent = 'Copied!';
            this.copyContent.style.background = '#48bb78';
            
            setTimeout(() => {
                this.copyContent.textContent = originalText;
                this.copyContent.style.background = '#667eea';
            }, 2000);
            
        } catch (error) {
            console.error('Failed to copy content:', error);
            this.showError('Failed to copy content to clipboard');
        }
    }

    downloadDocumentContent() {
        if (!this.currentModalFile) return;
        
        const content = this.currentModalFile.summary || '';
        const filename = `${this.currentModalFile.original_name || 'document'}_extracted.txt`;
        
        const blob = new Blob([content], { type: 'text/plain' });
        const url = URL.createObjectURL(blob);
        
        const a = document.createElement('a');
        a.href = url;
        a.download = filename;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        
        URL.revokeObjectURL(url);
    }

    // Enhanced sendMessage with improved error handling and reference display
    async sendMessage() {
        const message = this.chatInput.value.trim();
        if (!message || !this.sessionId || this.isProcessing) return;

        this.isProcessing = true;
        this.addMessage('user', message);
        this.chatInput.value = '';
        this.showTypingIndicator();

        try {
            console.log('Sending request:', {
                session_id: this.sessionId,
                question: message
            });

            const response = await fetch('/ask', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    session_id: this.sessionId,
                    question: message
                })
            });

            console.log('Response status:', response.status);

            if (!response.ok) {
                const errorText = await response.text();
                console.error('HTTP Error Response:', errorText);
                throw new Error(`Server returned ${response.status}: ${errorText}`);
            }

            const result = await response.json();
            console.log('Parsed JSON result:', result);
            
            this.hideTypingIndicator();

            if (result.status === 'success') {
                console.log('Processing successful response');
                
                // Validate required fields
                const answer = result.answer;
                const references = result.document_references || [];
                const filenames = result.referenced_filenames || [];
                
                console.log('Answer:', answer);
                console.log('References:', references);
                console.log('Filenames:', filenames);
                
                if (!answer) {
                    throw new Error('No answer received from server');
                }
                
                this.addMessage('bot', answer, {
                    confidence: result.confidence_score || 0,
                    references: references,
                    referencedFilenames: filenames,
                    sourcesUsed: result.sources_used || 0
                });
                
                // Enhanced highlighting: only highlight referenced documents
                if (references.length > 0) {
                    console.log('Highlighting referenced documents:', references);
                    this.highlightReferencedDocuments(references);
                }
                
            } else {
                console.error('Server returned error status:', result);
                throw new Error(result.error || 'Server returned error status');
            }
            
        } catch (error) {
            console.error('SendMessage error details:', error);
            console.error('Error stack:', error.stack);
            
            this.hideTypingIndicator();
            
            // Show more specific error message
            const errorMessage = error.message.includes('JSON') 
                ? 'Received invalid response from server. Please try again.'
                : `Error: ${error.message}`;
                
            this.addMessage('bot', `Sorry, I encountered an error: ${errorMessage}`);
            this.showError(error.message);
            
        } finally {
            this.isProcessing = false;
        }
    }

    // Fixed addMessage method that uses server-provided filenames
    addMessage(sender, content, metadata = {}) {
        const messageElement = document.createElement('div');
        messageElement.className = `message ${sender}-message`;
        
        let referencesHtml = '';
        // Use the filenames provided by the server response
        if (metadata.referencedFilenames && metadata.referencedFilenames.length > 0) {
            referencesHtml = `
                <div class="document-references">
                    <span class="references-label">📎 Referenced Documents:</span>
                    ${metadata.referencedFilenames.map(name => `<span class="doc-reference">${name}</span>`).join('')}
                </div>
            `;
        }

        let metaHtml = '';
        if (metadata.confidence !== undefined) {
            const confidencePercentage = Math.round(metadata.confidence * 100);
            metaHtml = `
                <div class="message-meta">
                    <span>Sources: ${metadata.sourcesUsed || 0}</span>
                    <span>Referenced: ${metadata.referencedFilenames ? metadata.referencedFilenames.length : 0}</span>
                    <span>Confidence: ${confidencePercentage}%</span>
                </div>
            `;
            this.updateConfidenceIndicator(confidencePercentage);
        }

        messageElement.innerHTML = `
            <div class="message-content">
                <p>${content}</p>
                ${referencesHtml}
                ${metaHtml}
            </div>
        `;

        this.chatMessages.appendChild(messageElement);
        setTimeout(() => {
            this.scrollToBottom();
        }, 100);
    }

    showTypingIndicator() {
        const typingElement = document.createElement('div');
        typingElement.className = 'message bot-message typing-indicator';
        typingElement.innerHTML = `
            <div class="message-content">
                <p>Thinking...</p>
            </div>
        `;
        this.chatMessages.appendChild(typingElement);
        this.scrollToBottom();
    }

    hideTypingIndicator() {
        const typingIndicator = this.chatMessages.querySelector('.typing-indicator');
        if (typingIndicator) {
            typingIndicator.remove();
        }
    }

    // Enhanced document highlighting - only highlights referenced documents
    highlightReferencedDocuments(references) {
        console.log('Highlighting documents:', references);
        
        // Remove previous highlights from all documents
        document.querySelectorAll('.document-item.referenced').forEach(doc => {
            doc.classList.remove('referenced');
        });

        // Only highlight documents that are actually referenced
        if (!references || references.length === 0) {
            console.log('No document references to highlight');
            return;
        }

        references.forEach(ref => {
            // Try exact match first
            let docElement = document.querySelector(`[data-doc-id="${ref}"]`);
            
            if (docElement) {
                console.log(`Highlighting document: ${ref}`);
                docElement.classList.add('referenced');
                
                // Scroll the referenced document into view
                docElement.scrollIntoView({ 
                    behavior: 'smooth', 
                    block: 'nearest' 
                });
                
                // Remove highlight after 8 seconds
                setTimeout(() => {
                    docElement.classList.remove('referenced');
                }, 8000);
            } else {
                console.warn(`Document element not found for reference: ${ref}`);
            }
        });
    }

    updateConfidenceIndicator(confidence) {
        this.confidenceIndicator.textContent = `Confidence: ${confidence}%`;
        
        // Color based on confidence level
        if (confidence >= 80) {
            this.confidenceIndicator.style.background = 'rgba(72, 187, 120, 0.3)';
        } else if (confidence >= 60) {
            this.confidenceIndicator.style.background = 'rgba(237, 137, 54, 0.3)';
        } else {
            this.confidenceIndicator.style.background = 'rgba(245, 101, 101, 0.3)';
        }
    }

    scrollToBottom() {
        if (this.chatMessages) {
            // Smooth scroll to bottom
            this.chatMessages.scrollTo({
                top: this.chatMessages.scrollHeight,
                behavior: 'smooth'
            });
        }
    }

    showUploadSection() {
        this.chatSection.style.display = 'none';
        this.uploadSection.style.display = 'block';
        this.uploadedFiles = [];
        this.displaySelectedFiles();
    }

    startNewSession() {
        this.sessionId = null;
        this.documents = [];
        this.uploadedFiles = [];
        this.sessionStatus.textContent = 'No active session';
        this.confidenceIndicator.textContent = '';
        this.confidenceIndicator.style.background = 'rgba(255, 255, 255, 0.2)';
        this.chatMessages.innerHTML = `
            <div class="message bot-message">
                <div class="message-content">
                    <p>Hello! Please upload your medical documents to get started. I can then answer questions about prescriptions, diagnoses, medications, and other medical information.</p>
                </div>
            </div>
        `;
        this.showUploadSection();
    }

    showLoading(text = 'Processing...') {
        this.loadingText.textContent = text;
        this.loadingOverlay.style.display = 'flex';
    }

    hideLoading() {
        this.loadingOverlay.style.display = 'none';
    }

    showError(message) {
        this.errorMessage.textContent = message;
        this.errorToast.style.display = 'flex';
        
        // Auto hide after 5 seconds
        setTimeout(() => {
            this.hideError();
        }, 5000);
    }

    hideError() {
        this.errorToast.style.display = 'none';
    }
}

// Initialize the application
let medicalChat;
document.addEventListener('DOMContentLoaded', () => {
    medicalChat = new MedicalDocumentChat();
});
