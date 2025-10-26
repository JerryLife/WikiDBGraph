import os
import pathlib

def create_symlinks():
    source_dir = "data/unzip"
    target_dir = "data/unziplink"
    
    # Create target directory if it doesn't exist
    os.makedirs(target_dir, exist_ok=True)
    
    # Get list of directories in source
    source_path = pathlib.Path(source_dir)
    for item in source_path.iterdir():
        if item.is_dir():
            # Create clean name by replacing spaces with hyphens
            clean_name = item.name.replace(" ", "-")
            target_path = pathlib.Path(target_dir) / clean_name
            
            # Create symbolic link
            if not target_path.exists():
                try:
                    os.symlink(item.resolve(), target_path)
                    print(f"Created symlink: {clean_name}")
                except OSError as e:
                    print(f"Error creating symlink for {item.name}: {e}")

if __name__ == "__main__":
    create_symlinks() 