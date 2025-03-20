import cv2
import numpy as np
import pytesseract
import re
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import os
import pandas as pd

# Muat model YOLO
# Pastikan model YOLO tersedia
model_path = os.path.join(os.path.dirname(__file__), "models")
weights_path = os.path.join(model_path, "yolov3.weights")
cfg_path = os.path.join(model_path, "yolov3.cfg")
coco_path = os.path.join(model_path, "coco.names")

# Cek keberadaan model
if not os.path.exists(weights_path) or not os.path.exists(cfg_path) or not os.path.exists(coco_path):
    messagebox.showerror("Error", "Model YOLO tidak ditemukan! Pastikan folder 'models' ada.")
    exit()
    
net = cv2.dnn.readNet(weights_path, cfg_path)
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

# Muat model kelas COCO
with open(coco_path, "r") as f:
    classes = [line.strip() for line in f.readlines()]

# List untuk menyimpan hasil OCR
results = []
failed_ocr = []

# Fungsi untuk memproses gambar dan melakukan OCR
def process_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        return None, "Gambar tidak dapat dibaca"

    height, width, channels = image.shape

    # Deteksi objek
    blob = cv2.dnn.blobFromImage(image, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    # Analisis deteksi
    class_ids = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.95 and classes[class_id] == "person":
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # Terapkan Non-Max Suppression
    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    if len(indices) == 0:
        return image, "Tidak terdeteksi"

    # Gambar bounding box dan ekstrak teks
    cleaned_numbers = []
    for i in indices.flatten():
        x, y, w, h = boxes[i]
        person_roi = image[y:y+h, x:x+w]

        if person_roi is None or person_roi.size == 0:
            continue

        gray = cv2.cvtColor(person_roi, cv2.COLOR_BGR2GRAY)
        custom_config = r'--psm 11'
        text = pytesseract.image_to_string(gray, config=custom_config)

        if text.strip():
            numbers_only = re.findall(r'\d{2,}(?:\.\d+)?', text)
            for number in numbers_only:
                if number not in cleaned_numbers:
                    cleaned_numbers.append(number)

    return image, ", ".join(cleaned_numbers) if cleaned_numbers else "Tidak terdeteksi"

# Fungsi untuk membuka folder dan memproses semua gambar
def open_folder():
    global results, failed_ocr
    results = []
    failed_ocr = []

    folder_path = filedialog.askdirectory()
    if not folder_path:
        return

    # Disable tombol saat proses berjalan
    open_button.config(state=tk.DISABLED)
    export_button.config(state=tk.DISABLED)
    report_button.config(state=tk.DISABLED)

    # Dapatkan daftar file gambar di folder
    image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))]
    total_images = len(image_files)
    if total_images == 0:
        messagebox.showinfo("Info", "Tidak ada gambar yang ditemukan di folder ini.")
        open_button.config(state=tk.NORMAL)
        return

    # Tampilkan loading bar
    loading_label.pack()
    root.update_idletasks()

    # Reset progress bar
    progress_bar["value"] = 0
    progress_bar["maximum"] = total_images
    root.update_idletasks()

    # Hapus data lama di tabel
    for row in result_tree.get_children():
        result_tree.delete(row)

    # Proses setiap gambar
    for idx, filename in enumerate(image_files):
        image_path = os.path.join(folder_path, filename)
        image, bib_numbers = process_image(image_path)

        # Simpan hasil ke list
        results.append((filename, bib_numbers))

        # Simpan log OCR yang gagal
        if bib_numbers == "Tidak terdeteksi":
            failed_ocr.append(filename)

        # Tambahkan hasil ke tabel
        result_tree.insert("", "end", values=(filename, bib_numbers))

        # Update progress bar
        progress_bar["value"] = idx + 1
        progress_label.config(text=f"Memproses {idx + 1}/{total_images} gambar...")
        root.update_idletasks()

    # Sembunyikan loading bar
    loading_label.pack_forget()
    progress_label.config(text="Proses selesai!")

    # Aktifkan kembali tombol
    open_button.config(state=tk.NORMAL)
    export_button.config(state=tk.NORMAL)
    report_button.config(state=tk.NORMAL)

# Fungsi untuk mengekspor hasil ke file CSV (hanya yang memiliki BIB Number)
def export_to_csv():
    file_path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV files", "*.csv")])
    if not file_path:
        return

    # Filter hasil hanya yang memiliki BIB Number
    filtered_results = [(filename, bib) for filename, bib in results if bib != "Tidak terdeteksi"]

    if not filtered_results:
        messagebox.showwarning("Peringatan", "Tidak ada data valid untuk diekspor!")
        return

    # Buat DataFrame
    df = pd.DataFrame(filtered_results, columns=["nama_image", "bibnumber"])

    # Simpan ke file CSV
    df.to_csv(file_path, index=False)
    messagebox.showinfo("Sukses", f"Data berhasil diekspor ke {file_path}")

# Fungsi untuk menampilkan log OCR yang gagal
def show_failed_ocr():
    if not failed_ocr:
        messagebox.showinfo("Info", "Tidak ada OCR yang gagal!")
        return

    # Buat jendela baru
    report_window = tk.Toplevel(root)
    report_window.title("Laporan OCR Gagal")

    # Tambahkan listbox untuk menampilkan file yang gagal OCR
    listbox = tk.Listbox(report_window, width=50, height=15)
    listbox.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)

    for filename in failed_ocr:
        listbox.insert(tk.END, filename)

# Buat GUI
root = tk.Tk()
root.title("Bibnumber OCR Detections")

frame = tk.Frame(root)
frame.pack(padx=10, pady=10)

open_button = tk.Button(frame, text="Buka Folder", command=open_folder)
open_button.pack(side=tk.LEFT, padx=5)

export_button = tk.Button(frame, text="Export ke CSV", command=export_to_csv, state=tk.DISABLED)
export_button.pack(side=tk.LEFT, padx=5)

report_button = tk.Button(frame, text="Lihat Report", command=show_failed_ocr, state=tk.DISABLED)
report_button.pack(side=tk.LEFT, padx=5)

progress_bar = ttk.Progressbar(root, orient="horizontal", length=400, mode="determinate")
progress_bar.pack(pady=10)

progress_label = tk.Label(root, text="")
progress_label.pack()

loading_label = tk.Label(root, text="Proses sedang berjalan...", fg="red")
loading_label.pack_forget()

columns = ("nama_image", "bibnumber")
result_tree = ttk.Treeview(root, columns=columns, show="headings")
result_tree.heading("nama_image", text="Nama Gambar")
result_tree.heading("bibnumber", text="BIB Number")
result_tree.pack(fill=tk.BOTH, expand=True)

root.mainloop()
