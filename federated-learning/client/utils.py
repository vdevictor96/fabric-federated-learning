import os

def get_file_path(relative_path):
    # Get the directory of the script
    script_dir = os.path.dirname(__file__)  # <-- absolute dir the script is in

    # Join the relative path with the script directory
    return os.path.abspath(os.path.join(script_dir, relative_path))