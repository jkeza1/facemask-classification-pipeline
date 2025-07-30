#!/usr/bin/env python3
"""
Face Mask Detection ML Pipeline - Main Entry Point
Pure Python implementation without React dependencies
"""

import os
import sys
import argparse
import logging
from datetime import datetime
import subprocess
import webbrowser
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def setup_directories():
    """Create necessary directories"""
    directories = [
        "models",
        "data",
        "logs", 
        "temp",
        "static",
        "notebooks"
    ]
    
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        logger.info(f"✅ Created directory: {directory}")

def install_dependencies():
    """Install required Python packages"""
    logger.info("📦 Installing Python dependencies...")
    
    packages = [
        "tensorflow>=2.10.0",
        "scikit-learn>=1.1.0", 
        "matplotlib>=3.5.0",
        "seaborn>=0.11.0",
        "pandas>=1.4.0",
        "numpy>=1.21.0",
        "pillow>=9.0.0",
        "opencv-python>=4.6.0",
        "fastapi>=0.85.0",
        "uvicorn[standard]>=0.18.0",
        "python-multipart>=0.0.5",
        "jupyter>=1.0.0",
        "ipykernel>=6.0.0"
    ]
    
    for package in packages:
        try:
            logger.info(f"Installing {package}...")
            subprocess.run([sys.executable, "-m", "pip", "install", package], 
                         check=True, capture_output=True, text=True)
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to install {package}: {e}")
            return False
    
    logger.info("✅ All dependencies installed successfully")
    return True

def check_dependencies():
    """Check if required packages are available"""
    logger.info("🔍 Checking dependencies...")
    
    required_modules = {
        'tensorflow': 'TensorFlow',
        'sklearn': 'scikit-learn', 
        'matplotlib': 'Matplotlib',
        'seaborn': 'Seaborn',
        'pandas': 'Pandas',
        'numpy': 'NumPy',
        'PIL': 'Pillow',
        'cv2': 'OpenCV',
        'fastapi': 'FastAPI',
        'uvicorn': 'Uvicorn'
    }
    
    missing = []
    for module, name in required_modules.items():
        try:
            __import__(module)
            logger.info(f"✅ {name} - OK")
        except ImportError:
            logger.warning(f"❌ {name} - Missing")
            missing.append(name)
    
    if missing:
        logger.error(f"Missing packages: {', '.join(missing)}")
        logger.info("Run: python main.py --install")
        return False
    
    logger.info("✅ All dependencies satisfied")
    return True

def run_training():
    """Start Jupyter notebook for model training"""
    logger.info("🧠 Starting model training...")
    
    notebook_path = "notebook/face_mask_detection_pipeline.ipynb"
    
    if not os.path.exists(notebook_path):
        logger.error(f"Training notebook not found: {notebook_path}")
        return False
    
    try:
        # Start Jupyter notebook
        logger.info("Opening Jupyter notebook for training...")
        subprocess.run(["jupyter", "notebook", notebook_path], check=True)
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to start Jupyter: {e}")
        return False
    except FileNotFoundError:
        logger.error("Jupyter not found. Install with: pip install jupyter")
        return False

def run_api_server():
    """Start FastAPI server"""
    logger.info("🚀 Starting FastAPI server...")
    
    server_script = "app/api/fastapi_server.py"
    
    if not os.path.exists(server_script):
        logger.error(f"FastAPI server not found: {server_script}")
        return False
    
    try:
        logger.info("FastAPI server starting at http://localhost:8000")
        logger.info("API documentation available at http://localhost:8000/docs")
        subprocess.run([sys.executable, server_script], check=True)
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to start API server: {e}")
        return False
    except KeyboardInterrupt:
        logger.info("API server stopped by user")
        return True

def open_web_interface():
    """Open the HTML web interface"""
    logger.info("🌐 Opening web interface...")
    
    html_file = "static/index.html"
    
    if not os.path.exists(html_file):
        logger.error(f"Web interface not found: {html_file}")
        return False
    
    try:
        # Open HTML file in default browser
        file_url = f"file://{os.path.abspath(html_file)}"
        webbrowser.open(file_url)
        logger.info(f"Web interface opened: {file_url}")
        return True
    except Exception as e:
        logger.error(f"Failed to open web interface: {e}")
        return False

def show_status():
    """Show current pipeline status"""
    logger.info("📊 Pipeline Status:")
    
    # Check if model exists
    model_files = list(Path("models").glob("*.h5"))
    if model_files:
        logger.info(f"✅ Trained models found: {len(model_files)}")
        for model in model_files:
            logger.info(f"   - {model.name}")
    else:
        logger.info("❌ No trained models found")
    
    # Check if data exists
    data_dir = Path("data")
    if data_dir.exists() and any(data_dir.iterdir()):
        logger.info("✅ Training data directory exists")
    else:
        logger.info("❌ No training data found")
    
    # Check if API server files exist
    api_files = ["app/api/fastapi_server.py", "static/index.html"]
    for file in api_files:
        if os.path.exists(file):
            logger.info(f"✅ {file} - Ready")
        else:
            logger.info(f"❌ {file} - Missing")

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Face Mask Detection ML Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --install          # Install dependencies
  python main.py --check            # Check dependencies  
  python main.py --train            # Start training
  python main.py --api              # Start API server
  python main.py --web              # Open web interface
  python main.py --status           # Show pipeline status
  python main.py --all              # Show complete instructions
        """
    )
    
    parser.add_argument("--install", action="store_true", 
                       help="Install required dependencies")
    parser.add_argument("--check", action="store_true",
                       help="Check if dependencies are installed")
    parser.add_argument("--train", action="store_true",
                       help="Start model training (Jupyter notebook)")
    parser.add_argument("--api", action="store_true", 
                       help="Start FastAPI server")
    parser.add_argument("--web", action="store_true",
                       help="Open web interface")
    parser.add_argument("--status", action="store_true",
                       help="Show pipeline status")
    parser.add_argument("--all", action="store_true",
                       help="Show complete pipeline instructions")
    
    args = parser.parse_args()
    
    # Setup directories
    setup_directories()
    
    if args.install:
        logger.info("🔧 Installing dependencies...")
        if install_dependencies():
            logger.info("✅ Installation completed successfully")
        else:
            logger.error("❌ Installation failed")
            sys.exit(1)
    
    elif args.check:
        if check_dependencies():
            logger.info("✅ All dependencies are ready")
        else:
            logger.error("❌ Missing dependencies")
            sys.exit(1)
    
    elif args.train:
        if not check_dependencies():
            logger.error("Dependencies missing. Run: python main.py --install")
            sys.exit(1)
        
        if run_training():
            logger.info("✅ Training session completed")
        else:
            logger.error("❌ Training failed")
            sys.exit(1)
    
    elif args.api:
        if not check_dependencies():
            logger.error("Dependencies missing. Run: python main.py --install")
            sys.exit(1)
        
        if run_api_server():
            logger.info("✅ API server session completed")
        else:
            logger.error("❌ API server failed")
            sys.exit(1)
    
    elif args.web:
        if open_web_interface():
            logger.info("✅ Web interface opened")
        else:
            logger.error("❌ Failed to open web interface")
            sys.exit(1)
    
    elif args.status:
        show_status()
    
    elif args.all:
        logger.info("🎭 Face Mask Detection ML Pipeline")
        logger.info("=" * 50)
        logger.info("")
        logger.info("📋 Complete Pipeline Steps:")
        logger.info("")
        logger.info("1️⃣  Install Dependencies:")
        logger.info("    python main.py --install")
        logger.info("")
        logger.info("2️⃣  Check Installation:")
        logger.info("    python main.py --check")
        logger.info("")
        logger.info("3️⃣  Prepare Your Dataset:")
        logger.info("    data/")
        logger.info("    ├── with_mask/")
        logger.info("    ├── without_mask/")
        logger.info("    └── mask_weared_incorrect/")
        logger.info("")
        logger.info("4️⃣  Train the Model:")
        logger.info("    python main.py --train")
        logger.info("    (Opens Jupyter notebook)")
        logger.info("")
        logger.info("5️⃣  Start API Server:")
        logger.info("    python main.py --api")
        logger.info("    (Available at http://localhost:8000)")
        logger.info("")
        logger.info("6️⃣  Open Web Interface:")
        logger.info("    python main.py --web")
        logger.info("    (Opens HTML interface)")
        logger.info("")
        logger.info("🔗 API Endpoints:")
        logger.info("    POST /predict        - Make predictions")
        logger.info("    GET  /model-status   - Check model status")
        logger.info("    POST /retrain        - Retrain model")
        logger.info("    GET  /docs           - API documentation")
        logger.info("")
        logger.info("📊 Check Status:")
        logger.info("    python main.py --status")
        
    else:
        parser.print_help()
        logger.info("\n🚀 Quick Start:")
        logger.info("python main.py --all    # Show complete instructions")
        logger.info("python main.py --install # Install dependencies")

if __name__ == "__main__":
    main()
