# gui/main_gui.py
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from evaluators.pytorch_evaluator import PyTorchEvaluator
from evaluators.tensorflow_evaluator import TFEvaluator
from evaluators.onnx_evaluator import ONNXEvaluator
from utils.dataset_loader import load_dataset
import os


def detect_framework(path):
    if path.endswith(('.pt', '.pth')):
        return "PyTorch"
    elif path.endswith(('.h5', '.pb')):
        return "TensorFlow"
    elif path.endswith('.onnx'):
        return "ONNX"
    return "Unknown"


def launch_gui():
    root = tk.Tk()
    root.title("CV Model Analyzer")
    root.geometry("500x400")

    model_path = tk.StringVar()
    framework = tk.StringVar()
    dataset = tk.StringVar(value="CIFAR-10")
    custom_dataset_path = tk.StringVar()
    image_width = tk.StringVar(value="224")
    image_height = tk.StringVar(value="224")

    def browse_model():
        path = filedialog.askopenfilename(
            title="Select Model File",
            filetypes=[
                ("PyTorch files", "*.pt *.pth"),
                ("TensorFlow files", "*.h5 *.pb"),
                ("ONNX files", "*.onnx"),
                ("All files", "*.*")
            ]
        )
        if path:
            model_path.set(path)
            framework.set(detect_framework(path))

    def browse_custom_dataset():
        path = filedialog.askdirectory(title="Select Custom Dataset Folder")
        if path:
            custom_dataset_path.set(path)

    def on_dataset_change(*args):
        if dataset.get() == "Custom":
            custom_frame.pack(pady=5)
        else:
            custom_frame.pack_forget()

    def run_evaluation():
        if not model_path.get() or framework.get() == "Unknown":
            messagebox.showerror("Error", "Please select a valid model.")
            return

        if dataset.get() == "Custom" and not custom_dataset_path.get():
            messagebox.showerror("Error", "Please select a custom dataset folder.")
            return

        try:
            # Get image size
            img_size = (int(image_width.get()), int(image_height.get()))

            # Load dataset
            if dataset.get() == "Custom":
                images, labels = load_dataset(dataset.get(), custom_dataset_path.get(), img_size)
            else:
                images, labels = load_dataset(dataset.get(), image_size=img_size)

            # Initialize evaluator
            evaluator = None
            if framework.get() == "PyTorch":
                evaluator = PyTorchEvaluator(model_path.get())
            elif framework.get() == "TensorFlow":
                evaluator = TFEvaluator(model_path.get())
            elif framework.get() == "ONNX":
                evaluator = ONNXEvaluator(model_path.get())

            # Run evaluation
            messagebox.showinfo("Info", f"Starting evaluation on {len(images)} images...")
            y_true, y_pred = evaluator.evaluate(images, labels)
            evaluator.show_metrics(y_true, y_pred)

        except Exception as e:
            messagebox.showerror("Error", f"Evaluation failed: {str(e)}")

    # Model selection
    ttk.Label(root, text="1. Select Model File", font=("Arial", 10, "bold")).pack(pady=5)
    model_frame = ttk.Frame(root)
    model_frame.pack(pady=5)
    ttk.Button(model_frame, text="Browse Model", command=browse_model).pack(side=tk.LEFT, padx=5)
    ttk.Label(model_frame, text="Framework:").pack(side=tk.LEFT, padx=5)
    ttk.Label(model_frame, textvariable=framework, foreground="blue").pack(side=tk.LEFT)

    # Dataset selection
    ttk.Label(root, text="2. Select Dataset", font=("Arial", 10, "bold")).pack(pady=(15, 5))
    dataset_frame = ttk.Frame(root)
    dataset_frame.pack(pady=5)
    datasets = ["CIFAR-10", "CIFAR-100", "Custom"]
    dataset_combo = ttk.Combobox(dataset_frame, textvariable=dataset, values=datasets, state="readonly")
    dataset_combo.pack()
    dataset.trace('w', on_dataset_change)

    # Custom dataset selection (hidden by default)
    custom_frame = ttk.Frame(root)
    custom_dataset_frame = ttk.Frame(custom_frame)
    custom_dataset_frame.pack(pady=5)
    ttk.Label(custom_dataset_frame, text="Custom Dataset Folder:").pack(side=tk.LEFT, padx=5)
    ttk.Button(custom_dataset_frame, text="Browse", command=browse_custom_dataset).pack(side=tk.LEFT, padx=5)
    ttk.Label(custom_frame, textvariable=custom_dataset_path, foreground="green", wraplength=400).pack(pady=2)

    # Image size settings
    ttk.Label(root, text="3. Image Size Settings", font=("Arial", 10, "bold")).pack(pady=(15, 5))
    size_frame = ttk.Frame(root)
    size_frame.pack(pady=5)
    ttk.Label(size_frame, text="Width:").pack(side=tk.LEFT, padx=5)
    ttk.Entry(size_frame, textvariable=image_width, width=8).pack(side=tk.LEFT, padx=2)
    ttk.Label(size_frame, text="Height:").pack(side=tk.LEFT, padx=5)
    ttk.Entry(size_frame, textvariable=image_height, width=8).pack(side=tk.LEFT, padx=2)

    # Run button
    ttk.Button(root, text="Run Evaluation", command=run_evaluation,
               style="Accent.TButton").pack(pady=20)

    # Instructions
    instructions = ttk.Label(root, text="Instructions:\n" +
                                        "• For custom datasets, organize images in folders by class\n" +
                                        "• Supported formats: JPG, PNG, BMP, TIFF\n" +
                                        "• PyTorch models should be saved with torch.save(model, path)",
                             justify=tk.LEFT, foreground="gray")
    instructions.pack(pady=10)

    root.mainloop()