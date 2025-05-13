document.addEventListener('DOMContentLoaded', () => {
    // DOM Elements
    const imageInput = document.getElementById('image-input');
    const preview = document.getElementById('preview');
    const uploadPlaceholder = document.getElementById('upload-placeholder');
    const removeImageBtn = document.getElementById('remove-image');
    const analyzeBtn = document.getElementById('analyze-btn');
    const uploadForm = document.getElementById('upload-form');
    const dropzone = document.getElementById('dropzone');
    const resultsSection = document.getElementById('results-section');
    const diseaseName = document.getElementById('disease-name');
    const confidenceBar = document.getElementById('confidence-progress');
    const confidenceValue = document.getElementById('confidence-value');
    const treatmentInfo = document.getElementById('treatment-info');
    const newScanBtn = document.getElementById('new-scan');
    const loader = document.getElementById('loader');
    const aboutTab = document.getElementById('about-tab');
    const browseBtn = document.getElementById('browse-btn');
    
    // Navigate to About page
    if (aboutTab) {
        aboutTab.addEventListener('click', () => {
            window.location.href = 'about.html';
        });
    }
    
    // File upload handling
    if (imageInput) {
        imageInput.addEventListener('change', handleImageSelect);
    }

    // Enhanced Browse button functionality
    if (browseBtn) {
        browseBtn.addEventListener('click', function(event) {
            event.preventDefault();
            event.stopPropagation();
            if (imageInput) {
                imageInput.click();
            }
        });
    }
    
    // Make the entire dropzone clickable
    if (dropzone) {
        dropzone.addEventListener('click', function(e) {
            if (e.target === uploadPlaceholder || e.target === dropzone) {
                imageInput.click();
            }
        });
        
        // Drag and drop handling
        ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
            dropzone.addEventListener(eventName, preventDefaults, false);
        });
        
        ['dragenter', 'dragover'].forEach(eventName => {
            dropzone.addEventListener(eventName, highlight, false);
        });
        
        ['dragleave', 'drop'].forEach(eventName => {
            dropzone.addEventListener(eventName, unhighlight, false);
        });
        
        dropzone.addEventListener('drop', handleDrop, false);
    }
    
    function preventDefaults(e) {
        e.preventDefault();
        e.stopPropagation();
    }
    
    function highlight() {
        dropzone.classList.add('dragover');
    }
    
    function unhighlight() {
        dropzone.classList.remove('dragover');
    }
    
    function handleDrop(e) {
        const dt = e.dataTransfer;
        const files = dt.files;
        
        if (files.length) {
            imageInput.files = files;
            handleImageSelect();
        }
    }
    
    function handleImageSelect() {
        const file = imageInput.files[0];
        
        if (file) {
            const reader = new FileReader();
            
            reader.onload = function(e) {
                preview.src = e.target.result;
                preview.classList.remove('hidden');
                uploadPlaceholder.classList.add('hidden');
                removeImageBtn.classList.remove('hidden');
                analyzeBtn.disabled = false;
            };
            
            reader.readAsDataURL(file);
        }
    }
    
    // Remove image button
    if (removeImageBtn) {
        removeImageBtn.addEventListener('click', () => {
            resetImageUpload();
        });
    }
    
    function resetImageUpload() {
        imageInput.value = '';
        preview.src = '';
        preview.classList.add('hidden');
        uploadPlaceholder.classList.remove('hidden');
        removeImageBtn.classList.add('hidden');
        analyzeBtn.disabled = true;
        if (resultsSection) {
            resultsSection.classList.add('hidden');
        }
    }
    
    // Form submission and API call
    if (uploadForm) {
        uploadForm.addEventListener('submit', async function(e) {
            e.preventDefault();
            
            const file = imageInput.files[0];
            if (!file) return;
            
            // Show loader
            if (loader) {
                loader.classList.remove('hidden');
            }
            
            const formData = new FormData();
            formData.append('image', file);
            
            try {
                const response = await fetch('http://127.0.0.1:5000/predict', {
                    method: 'POST',
                    body: formData
                });
                
                const data = await response.json();
                
                // Hide loader
                if (loader) {
                    loader.classList.add('hidden');
                }
                
                if (data.success) {
                    // Only proceed if all required elements exist
                    if (diseaseName && confidenceBar && confidenceValue && treatmentInfo && resultsSection) {
                        // Display results
                        diseaseName.textContent = data.disease;
                        
                        // Animate confidence bar
                        const confidence = data.confidence;
                        confidenceValue.textContent = `${confidence}%`;
                        
                        // Small delay for animation effect
                        setTimeout(() => {
                            confidenceBar.style.width = `${confidence}%`;
                            
                            // Set color based on confidence level
                            if (confidence < 50) {
                                confidenceBar.style.background = 'linear-gradient(90deg, #ff5252, #ff7752)';
                            } else if (confidence < 80) {
                                confidenceBar.style.background = 'linear-gradient(90deg, #ffb74d, #ffa726)';
                            }
                            // Default green color for high confidence is set in CSS
                        }, 100);
                        
                        treatmentInfo.textContent = data.treatment;
                        
                        // Show results section
                        resultsSection.classList.remove('hidden');
                    } else {
                        console.error("Required result elements not found in the DOM");
                        alert('UI Error: Could not display results. See console for details.');
                    }
                } else {
                    alert('Error: ' + data.error);
                }
            } catch (err) {
                if (loader) {
                    loader.classList.add('hidden');
                }
                alert('Server error. Please try again later.');
                console.error(err);
            }
        });
    }
    
    // New scan button
    if (newScanBtn) {
        newScanBtn.addEventListener('click', () => {
            resetImageUpload();
        });
    }
});