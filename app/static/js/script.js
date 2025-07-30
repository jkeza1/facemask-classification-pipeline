
document.addEventListener('DOMContentLoaded', function () {
    // DOM Elements
    const uploadArea = document.getElementById('uploadArea');
    const fileInput = document.getElementById('fileInput');
    const predictBtn = document.getElementById('predictBtn');
    const uploadLabel = document.querySelector('.upload-label');
    const uploadForm = document.getElementById('uploadForm');
    const spinner = document.getElementById('spinner');

    // Click to open file dialog
    uploadArea.addEventListener('click', () => fileInput.click());

    // File input change handler
    fileInput.addEventListener('change', () => {
        if (fileInput.files.length) {
            predictBtn.disabled = false;
            uploadArea.classList.add('highlight');
            uploadLabel.textContent = fileInput.files[0].name;
        }
    });

    // Drag and drop events
    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
        uploadArea.addEventListener(eventName, preventDefaults, false);
    });

    ['dragenter', 'dragover'].forEach(eventName => {
        uploadArea.addEventListener(eventName, highlight, false);
    });

    ['dragleave', 'drop'].forEach(eventName => {
        uploadArea.addEventListener(eventName, unhighlight, false);
    });

    uploadArea.addEventListener('drop', (e) => {
        const dt = e.dataTransfer;
        const files = dt.files;
        fileInput.files = files;
        predictBtn.disabled = false;
        uploadArea.classList.add('highlight');
        uploadLabel.textContent = files[0].name;
    });

    // Show spinner on submit
    uploadForm.addEventListener('submit', () => {
        spinner.style.display = 'block';
    });

    // Utility functions
    function preventDefaults(e) {
        e.preventDefault();
        e.stopPropagation();
    }

    function highlight() {
        uploadArea.classList.add('highlight');
    }

    function unhighlight() {
        uploadArea.classList.remove('highlight');
    }
});

