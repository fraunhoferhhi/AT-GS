import os
import shutil

def delete_folders_with_prefix(directory, prefix="normal"):
    for root, dirs, _ in os.walk(directory, topdown=False):
        for folder in dirs:
            if folder.startswith(prefix):
                folder_path = os.path.join(root, folder)
                try:
                    shutil.rmtree(folder_path)
                    print(f"Deleted folder: {folder_path}")
                except Exception as e:
                    print(f"Failed to delete folder: {folder_path}. Error: {e}")

directory_path = '/media/dc-04-vol03/HBR/storage/P23002_Geisha/14_colmap/T015'
delete_folders_with_prefix(directory_path)
