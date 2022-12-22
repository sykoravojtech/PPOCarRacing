import os
import sys
import shutil                

def move_tensorboard_files_to(from_dir, to_dir="tensorboard/"):
    if not os.path.exists(to_dir):
        os.makedirs(to_dir)
    
    folders = os.listdir(from_dir)
    for folder in folders:
        print(f"copying from '{folder}'")
        tboard_path = os.path.join(from_dir, folder, "tboard")
        filepath = os.path.join(tboard_path, os.listdir(tboard_path)[0])
        filename = os.path.basename(filepath)
        print(f"{filename=}")
        os.makedirs(os.path.join(to_dir, folder))
        shutil.copyfile(filepath, os.path.join(to_dir, folder, filename))

if len(sys.argv) > 1:
    print(f"folder = {sys.argv[1]}")
else:
    print("NO DIRECTORY INPUTTED")
    exit()
    
move_tensorboard_files_to(sys.argv[1])
