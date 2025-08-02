// Run when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
  // DOM Elements
  const uploadArea = document.getElementById('uploadArea');
  const fileInput = document.getElementById('fileInput');
  const predictBtn = document.getElementById('predictBtn');
  const uploadLabel = document.querySelector('.upload-label');
  const uploadForm = document.getElementById('uploadForm');
  const spinner = document.getElementById('spinner');

  uploadArea.addEventListener('click', () => fileInput.click());

  fileInput.addEventListener('change', () => handleFileSelection(fileInput.files));

  ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(event =>
    uploadArea.addEventListener(event, preventDefaults, false)
  );

  ['dragenter', 'dragover'].forEach(event =>
    uploadArea.addEventListener(event, () => highlight(uploadArea), false)
  );

  ['dragleave', 'drop'].forEach(event =>
    uploadArea.addEventListener(event, () => unhighlight(uploadArea), false)
  );

  uploadArea.addEventListener('drop', e => {
    const files = e.dataTransfer.files;
    fileInput.files = files;
    handleFileSelection(files);
  });

  uploadForm.addEventListener('submit', () => {
    spinner.style.display = 'block';
  });

  function preventDefaults(e) {
    e.preventDefault();
    e.stopPropagation();
  }

  function highlight(el) {
    el.classList.add('highlight');
  }

  function unhighlight(el) {
    el.classList.remove('highlight');
  }

  function handleFileSelection(files) {
    if (files.length) {
      predictBtn.disabled = false;
      highlight(uploadArea);
      uploadLabel.textContent = files[0].name;
    }
  }

  // Handle retrain success messages
  const message = document.querySelector('.retrain-result');
  const success = document.querySelector('.success-message');

  if (message || success) {
    setTimeout(() => {
      if (message) message.classList.add('slide-fade-out');
      if (success) success.classList.add('slide-fade-out');

      // Hide after animation
      setTimeout(() => {
        if (message) message.style.display = 'none';
        if (success) success.style.display = 'none';
      }, 500); // Match this to your CSS animation duration
    }, 4000);
  }
});