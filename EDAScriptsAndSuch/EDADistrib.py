import os
import matplotlib.pyplot as plt

def count_files_in_directory(directory_path):
    folder_names = []
    file_counts = []

    for folder in os.listdir(directory_path):
        folder_path = os.path.join(directory_path, folder)
        if os.path.isdir(folder_path):
            folder_names.append(folder)
            file_count = len([f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))])            
            file_counts.append(file_count)

    return folder_names, file_counts

def create_chart(folder_names, file_counts):
    plt.figure(figsize=(10, 6))
    plt.bar(folder_names, file_counts)
    plt.xlabel('Subfolders')
    plt.ylabel('Number of Files')
    plt.title('Number of Files in Each Subfolder')
    plt.xticks(rotation=60, ha="center")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    directory_path = input("Enter the directory path: ")

    if not os.path.exists(directory_path) or not os.path.isdir(directory_path):
        print("Invalid directory path. Please provide a valid directory path.")
    else:
        folder_names, file_counts = count_files_in_directory(directory_path)
        create_chart(folder_names, file_counts)