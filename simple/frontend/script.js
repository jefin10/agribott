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
    const confidenceBar = document.getElementById('confidence-bar');
    const confidenceValue = document.getElementById('confidence-value');
    const treatmentInfo = document.getElementById('treatment-info');
    const newScanBtn = document.getElementById('new-scan');
    const loader = document.getElementById('loader');
    const aboutTab = document.getElementById('about-tab');
    const browseBtn = document.getElementById('browse-btn');
    
    // Navigate to About page
    aboutTab.addEventListener('click', () => {
        window.location.href = 'about.html';
    });
    
    // File upload handling
    imageInput.addEventListener('change', handleImageSelect);

    // Enhanced Browse button functionality
    if (browseBtn) {
        browseBtn.addEventListener('click', function(event) {
            event.preventDefault(); // Prevent default button action
            event.stopPropagation(); // Stop event from bubbling up to dropzone
            if (imageInput) {
                imageInput.click();     // Trigger click on the hidden file input
            } else {
                console.error("Image input element (image-input) not found.");
            }
        });
    } else {
        console.error("Browse button (browse-btn) not found.");
    }
    
    // Make the entire dropzone clickable (optional enhancement)
    dropzone.addEventListener('click', function(e) {
        // Only trigger if clicking directly on the placeholder area
        if (e.target === uploadPlaceholder || e.target === dropzone) {
            imageInput.click();
        }
    });
    
    // Drag and drop handling
    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
        dropzone.addEventListener(eventName, preventDefaults, false);
    });
    
    function preventDefaults(e) {
        e.preventDefault();
        e.stopPropagation();
    }
    
    ['dragenter', 'dragover'].forEach(eventName => {
        dropzone.addEventListener(eventName, highlight, false);
    });
    
    ['dragleave', 'drop'].forEach(eventName => {
        dropzone.addEventListener(eventName, unhighlight, false);
    });
    
    function highlight() {
        dropzone.classList.add('dragover');
    }
    
    function unhighlight() {
        dropzone.classList.remove('dragover');
    }
    
    dropzone.addEventListener('drop', handleDrop, false);
    
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
    removeImageBtn.addEventListener('click', () => {
        resetImageUpload();
    });
    
    function resetImageUpload() {
        imageInput.value = '';
        preview.src = '';
        preview.classList.add('hidden');
        uploadPlaceholder.classList.remove('hidden');
        removeImageBtn.classList.add('hidden');
        analyzeBtn.disabled = true;
        resultsSection.classList.add('hidden');
    }
    
    // Form submission and API call
    uploadForm.addEventListener('submit', async function(e) {
        e.preventDefault();
        
        const file = imageInput.files[0];
        if (!file) return;
        
        // Show loader
        loader.classList.remove('hidden');
        
        const formData = new FormData();
        formData.append('image', file);
        
        try {
            const response = await fetch('http://127.0.0.1:5000/predict', {
                method: 'POST',
                body: formData
            });
            
            const data = await response.json();
            
            // Hide loader
            loader.classList.add('hidden');
            
            if (data.success) {
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
                alert('Error: ' + data.error);
            }
        } catch (err) {
            loader.classList.add('hidden');
            alert('Server error. Please try again later.');
            console.error(err);
        }
    });
    
    // New scan button
    newScanBtn.addEventListener('click', () => {
        resetImageUpload();
    });
});