"""
Diagnostic script to help understand directory structure for the fake news detector.
"""
import os
import sys

def print_directory_structure(start_path):
    """
    Print the directory structure starting from the given path
    """
    print(f"\nDirectory structure from {start_path}:")
    for root, dirs, files in os.walk(start_path):
        level = root.replace(start_path, '').count(os.sep)
        indent = ' ' * 4 * level
        print(f"{indent}{os.path.basename(root)}/")
        sub_indent = ' ' * 4 * (level + 1)
        for file in files:
            print(f"{sub_indent}{file}")

def check_model_paths():
    """
    Check where the model files should be located
    """
    # Get the current directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    print(f"Current directory: {current_dir}")
    
    # Get the parent directory
    parent_dir = os.path.dirname(current_dir)
    print(f"Parent directory: {parent_dir}")
    
    # Construct model directory path
    model_dir = os.path.join(parent_dir, 'models')
    print(f"Expected model directory: {model_dir}")
    
    # Check if model directory exists
    if os.path.exists(model_dir):
        print(f"Model directory exists: {model_dir}")
        # Check for model files
        model_path = os.path.join(model_dir, 'fake_news_classifier.pkl')
        vectorizer_path = os.path.join(model_dir, 'tfidf_vectorizer.pkl')
        
        if os.path.exists(model_path):
            print(f"✅ Model file exists: {model_path}")
            print(f"   Size: {os.path.getsize(model_path) / 1024:.2f} KB")
        else:
            print(f"❌ Model file NOT found: {model_path}")
        
        if os.path.exists(vectorizer_path):
            print(f"✅ Vectorizer file exists: {vectorizer_path}")
            print(f"   Size: {os.path.getsize(vectorizer_path) / 1024:.2f} KB")
        else:
            print(f"❌ Vectorizer file NOT found: {vectorizer_path}")
    else:
        print(f"❌ Model directory NOT found: {model_dir}")
        print(f"Creating model directory: {model_dir}")
        os.makedirs(model_dir, exist_ok=True)
        print(f"✅ Model directory created: {model_dir}")
    
    # Print the directory structure
    print_directory_structure(parent_dir)
    
    return model_dir

if __name__ == "__main__":
    model_dir = check_model_paths()
    print(f"\nTo fix your issue, place the model files in: {model_dir}")
    print("Run the generate_model.py script in the correct directory to create them.")