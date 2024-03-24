import tkinter as tk
from tkinter import filedialog
from tkinter import messagebox
from threading import Thread
from face_datasets import collect_training_data
from training import train_model

class FaceRecognitionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Face Recognition App")

        self.name_label = tk.Label(root, text="Enter Your Name:")
        self.name_entry = tk.Entry(root)
        self.start_button = tk.Button(root, text="Start Face Dataset Collection", command=self.start_dataset_collection)
        self.train_button = tk.Button(root, text="Train Model", command=self.train_model)
        self.browse_button = tk.Button(root, text="Browse Video File", command=self.browse_video_file)
        self.recognize_button = tk.Button(root, text="Start Face Recognition", command=self.start_face_recognition)
        self.video_path = ""

        self.name_label.pack(pady=10)
        self.name_entry.pack(pady=10)
        self.start_button.pack(pady=10)
        self.train_button.pack(pady=10)
        self.browse_button.pack(pady=10)
        self.recognize_button.pack(pady=10)

    def start_dataset_collection(self):
        name = self.name_entry.get()
        if not name:
            messagebox.showinfo("Error", "Please enter your name.")
            return

        collect_thread = Thread(target=self.collect_data_thread, args=(name,))
        collect_thread.start()

    def collect_data_thread(self, name):
        collect_training_data(name)
        messagebox.showinfo("Success", f"Dataset for {name} created successfully.")

    def train_model(self):
        train_thread = Thread(target=self.train_model_thread)
        train_thread.start()

    def train_model_thread(self):
        train_model()
        messagebox.showinfo("Success", "Model training completed successfully.")

    def browse_video_file(self):
        self.video_path = filedialog.askopenfilename(filetypes=[("Video files", "*.mp4;*.avi")])

    def start_face_recognition(self):
        if not self.video_path:
            messagebox.showinfo("Error", "Please browse a video file first.")
            return

        recognize_thread = Thread(target=self.recognize_faces_thread, args=(self.video_path,))
        recognize_thread.start()

    def recognize_faces_thread(self, video_path):
        # Call the function for face recognition here
        pass


if __name__ == "__main__":
    root = tk.Tk()
    app = FaceRecognitionApp(root)
    root.mainloop()
