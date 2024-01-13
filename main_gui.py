import tkinter as tk
from tkinter import filedialog, ttk, messagebox

class CameraRecognition:
    def __init__(self, parent):
        self.parent = parent
        self.process_running = False
        self.create_widgets()

    def create_widgets(self):
        label = tk.Label(self.parent, text="Face Recognition (Camera) Content")
        label.pack(pady=10)

        start_button = tk.Button(self.parent, text="Start Recognition", command=self.start_camera_recognition)
        start_button.pack(pady=10)

    def start_camera_recognition(self):
        self.process_running = True
        try:
            import face_recognition_main
            face_recognition_main.start_camera_recognition()
        finally:
            self.process_running = False

    def interrupt_process(self):
        if self.process_running:
            # Add code here to interrupt the camera recognition process
            messagebox.showinfo("Interrupt", "Camera Recognition process interrupted.")
        else:
            messagebox.showinfo("No Process Running", "No recognition process is currently running.")

class VideoRecognition:
    def __init__(self, parent):
        self.parent = parent
        self.video_file = ""
        self.process_running = False
        self.create_widgets()

    def create_widgets(self):
        label = tk.Label(self.parent, text="Face Recognition (Video) Content")
        label.pack(pady=10)

        browse_button = tk.Button(self.parent, text="Browse Video", command=self.browse_video)
        browse_button.pack(pady=10)

        start_button = tk.Button(self.parent, text="Start Recognition", command=self.start_recognition_video)
        start_button.pack(pady=10)

    def browse_video(self):
        self.video_file = filedialog.askopenfilename(filetypes=[("MP4 Files", "*.mp4")])
        if self.video_file:
            messagebox.showinfo("Video Selected", f"Selected video: {self.video_file}")

    def start_recognition_video(self):
        self.process_running = True
        try:
            import face_recognition_from_video
            face_recognition_from_video.recognize_faces_from_video(self.video_file)
        finally:
            self.process_running = False

    def interrupt_process(self):
        if self.process_running:
            # Add code here to interrupt the video recognition process
            messagebox.showinfo("Interrupt", "Video Recognition process interrupted.")
        else:
            messagebox.showinfo("No Process Running", "No recognition process is currently running.")

class ImageCapture:
    def __init__(self, parent):
        self.parent = parent
        self.process_running = False
        self.create_widgets()

    def create_widgets(self):
        label = tk.Label(self.parent, text="Capture Images for Training Content")
        label.pack(pady=10)

        capture_button = tk.Button(self.parent, text="Capture Images", command=self.capture_images)
        capture_button.pack(pady=10)

    def capture_images(self):
        self.process_running = True
        try:
            import face_datasets
            face_datasets.capture_images()
        finally:
            self.process_running = False

    def interrupt_process(self):
        if self.process_running:
            # Add code here to interrupt the image capture process
            messagebox.showinfo("Interrupt", "Image Capture process interrupted.")
        else:
            messagebox.showinfo("No Process Running", "No recognition process is currently running.")

class ModelTraining:
    def __init__(self, parent):
        self.parent = parent
        self.process_running = False
        self.create_widgets()

    def create_widgets(self):
        label = tk.Label(self.parent, text="Model Training Content")
        label.pack(pady=10)

        train_button = tk.Button(self.parent, text="Train Model", command=self.train_model)
        train_button.pack(pady=10)

    def train_model(self):
        self.process_running = True
        try:
            import training
            training.train_model()
        finally:
            self.process_running = False

    def interrupt_process(self):
        if self.process_running:
            # Add code here to interrupt the model training process
            messagebox.showinfo("Interrupt", "Model Training process interrupted.")
        else:
            messagebox.showinfo("No Process Running", "No recognition process is currently running.")

class FaceRecognitionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Face Recognition App")

        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill='both', expand=True)

        self.tab1 = ttk.Frame(self.notebook)
        self.tab2 = ttk.Frame(self.notebook)
        self.tab3 = ttk.Frame(self.notebook)
        self.tab4 = ttk.Frame(self.notebook)

        self.notebook.add(self.tab1, text='Face Recognition (Camera)')
        self.notebook.add(self.tab2, text='Face Recognition (Video)')
        self.notebook.add(self.tab3, text='Capture Images for Training')
        self.notebook.add(self.tab4, text='Model Training')

        self.create_menu()
        CameraRecognition(self.tab1)
        VideoRecognition(self.tab2)
        ImageCapture(self.tab3)
        ModelTraining(self.tab4)

        self.root.bind('<KeyPress-q>', self.interrupt_processes)

    def create_menu(self):
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)

        interrupt_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Interrupt", menu=interrupt_menu)
        interrupt_menu.add_command(label="Interrupt Camera Recognition", command=self.interrupt_camera_recognition)
        interrupt_menu.add_command(label="Interrupt Video Recognition", command=self.interrupt_video_recognition)
        interrupt_menu.add_command(label="Interrupt Image Capture", command=self.interrupt_image_capture)
        interrupt_menu.add_command(label="Interrupt Model Training", command=self.interrupt_model_training)

    def interrupt_camera_recognition(self):
        self.tab1.winfo_children()[0].interrupt_process()

    def interrupt_video_recognition(self):
        self.tab2.winfo_children()[0].interrupt_process()

    def interrupt_image_capture(self):
        self.tab3.winfo_children()[0].interrupt_process()

    def interrupt_model_training(self):
        self.tab4.winfo_children()[0].interrupt_process()

    def interrupt_processes(self, event):
        self.interrupt_camera_recognition()
        self.interrupt_video_recognition()
        self.interrupt_image_capture()
        self.interrupt_model_training()

def main():
    root = tk.Tk()
    app = FaceRecognitionApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()
