document.addEventListener('DOMContentLoaded', () => {
    // Navigate to Detect Disease page
    const detectTab = document.getElementById('detect-tab');
    
    detectTab.addEventListener('click', () => {
        window.location.href = 'index.html';
    });
});