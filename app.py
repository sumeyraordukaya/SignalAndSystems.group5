import os
import sys
import traceback
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import pandas as pd
import librosa

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from feature_extraction import extract_features
from f0_estimation import compute_autocorrelation_f0
from classifier import classify_gender_from_f0
from main import load_metadata, process_audio_files

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(BASE_DIR, "results")


class SpeechApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Speech Signal Analysis and Gender Classification")
        self.root.geometry("980x700")
        self.root.configure(bg="#f4f6f8")

        self.build_ui()

    def build_ui(self):
        title = tk.Label(
            self.root,
            text="Speech Signal Analysis and Gender Classification",
            font=("Arial", 18, "bold"),
            bg="#f4f6f8",
            fg="#1f2937"
        )
        title.pack(pady=15)

        subtitle = tk.Label(
            self.root,
            text="Single File Prediction and Dataset Performance",
            font=("Arial", 11),
            bg="#f4f6f8",
            fg="#4b5563"
        )
        subtitle.pack(pady=(0, 15))

        top_frame = tk.Frame(self.root, bg="#f4f6f8")
        top_frame.pack(fill="x", padx=20, pady=10)

        btn_select = tk.Button(
            top_frame,
            text="Select WAV File",
            command=self.select_file,
            font=("Arial", 11, "bold"),
            bg="#2563eb",
            fg="white",
            width=18,
            relief="flat",
            padx=10,
            pady=8
        )
        btn_select.pack(side="left", padx=(0, 10))

        btn_dataset = tk.Button(
            top_frame,
            text="Run Full Dataset Analysis",
            command=self.run_dataset_analysis,
            font=("Arial", 11, "bold"),
            bg="#059669",
            fg="white",
            width=22,
            relief="flat",
            padx=10,
            pady=8
        )
        btn_dataset.pack(side="left")

        result_frame = tk.LabelFrame(
            self.root,
            text="Single File Prediction",
            font=("Arial", 12, "bold"),
            bg="#f4f6f8",
            fg="#111827",
            padx=10,
            pady=10
        )
        result_frame.pack(fill="x", padx=20, pady=10)

        self.file_label = tk.Label(
            result_frame,
            text="No file selected.",
            font=("Arial", 10),
            bg="#f4f6f8",
            fg="#374151",
            anchor="w",
            justify="left"
        )
        self.file_label.pack(fill="x", pady=5)

        self.prediction_text = tk.Text(
            result_frame,
            height=8,
            font=("Arial", 11),
            bg="white",
            fg="#111827",
            wrap="word"
        )
        self.prediction_text.pack(fill="x", pady=5)
        self.prediction_text.insert("1.0", "Prediction results will appear here.")
        self.prediction_text.config(state="disabled")

        table_frame = tk.LabelFrame(
            self.root,
            text="Dataset Performance",
            font=("Arial", 12, "bold"),
            bg="#f4f6f8",
            fg="#111827",
            padx=10,
            pady=10
        )
        table_frame.pack(fill="both", expand=True, padx=20, pady=10)

        columns = (
            "Class",
            "Number of Samples",
            "Average F0 (Hz)",
            "Standard Deviation",
            "Success (%)"
        )

        self.tree = ttk.Treeview(table_frame, columns=columns, show="headings", height=8)

        for col in columns:
            self.tree.heading(col, text=col)
            self.tree.column(col, width=160, anchor="center")

        self.tree.pack(fill="x", pady=10)

        cm_label = tk.Label(
            table_frame,
            text="Confusion Matrix",
            font=("Arial", 11, "bold"),
            bg="#f4f6f8",
            fg="#111827"
        )
        cm_label.pack(anchor="w", pady=(10, 5))

        self.cm_text = tk.Text(
            table_frame,
            height=10,
            font=("Courier", 11),
            bg="white",
            fg="#111827",
            wrap="none"
        )
        self.cm_text.pack(fill="both", expand=True)
        self.cm_text.insert("1.0", "Confusion matrix will appear here.")
        self.cm_text.config(state="disabled")

    def update_prediction_box(self, text):
        self.prediction_text.config(state="normal")
        self.prediction_text.delete("1.0", tk.END)
        self.prediction_text.insert("1.0", text)
        self.prediction_text.config(state="disabled")

    def update_cm_box(self, text):
        self.cm_text.config(state="normal")
        self.cm_text.delete("1.0", tk.END)
        self.cm_text.insert("1.0", text)
        self.cm_text.config(state="disabled")

    def select_file(self):
        file_path = filedialog.askopenfilename(
            title="Select a WAV file",
            filetypes=[("WAV files", "*.wav")]
        )

        if not file_path:
            return

        self.file_label.config(text=f"Selected File: {file_path}")

        try:
            audio, sr = librosa.load(file_path, sr=None)

            energy, zcr, voiced_frames = extract_features(audio, sr)
            avg_f0 = compute_autocorrelation_f0(audio, sr)
            predicted_gender = classify_gender_from_f0(avg_f0)

            avg_energy = float(energy.mean()) if energy is not None and len(energy) > 0 else 0.0
            avg_zcr = float(zcr.mean()) if zcr is not None and len(zcr) > 0 else 0.0
            voiced_count = int(voiced_frames.sum()) if voiced_frames is not None else 0

            result_text = (
                f"Predicted Class: {predicted_gender.capitalize()}\n"
                f"Average F0 (Hz): {avg_f0:.2f}\n"
                f"Average Energy: {avg_energy:.4f}\n"
                f"Average ZCR: {avg_zcr:.4f}\n"
                f"Voiced Frame Count: {voiced_count}"
            )

            self.update_prediction_box(result_text)

        except Exception as e:
            messagebox.showerror("Error", f"File could not be processed.\n\n{e}")
            self.update_prediction_box("Prediction failed.")
            print(traceback.format_exc())

    def run_dataset_analysis(self):
        try:
            df = load_metadata()

            if df is None or df.empty:
                messagebox.showerror("Error", "Metadata could not be loaded.")
                return

            results_df = process_audio_files(df)

            if results_df is None or results_df.empty:
                messagebox.showerror("Error", "No results were produced.")
                return

            valid_df = results_df[
                results_df["actual_gender"].isin(["male", "female", "child"])
            ].copy()

            for item in self.tree.get_children():
                self.tree.delete(item)

            performance_rows = []

            for cls in ["male", "female", "child"]:
                subset = valid_df[valid_df["actual_gender"] == cls]

                if len(subset) == 0:
                    continue

                n_samples = len(subset)
                avg_f0 = subset["avg_f0"].mean()
                std_f0 = subset["avg_f0"].std()
                correct = (subset["actual_gender"] == subset["predicted_gender"]).sum()
                success = (correct / n_samples) * 100

                row = (
                    cls.capitalize(),
                    n_samples,
                    round(avg_f0, 2),
                    round(std_f0, 2) if pd.notna(std_f0) else 0.0,
                    round(success, 2)
                )

                performance_rows.append(row)
                self.tree.insert("", tk.END, values=row)

            confusion_matrix = pd.crosstab(
                valid_df["actual_gender"],
                valid_df["predicted_gender"]
            ).reindex(
                index=["male", "female", "child"],
                columns=["male", "female", "child"],
                fill_value=0
            )

            self.update_cm_box(confusion_matrix.to_string())

            os.makedirs(RESULTS_DIR, exist_ok=True)

            perf_df = pd.DataFrame(
                performance_rows,
                columns=[
                    "Class",
                    "Number of Samples",
                    "Average F0 (Hz)",
                    "Standard Deviation",
                    "Success (%)"
                ]
            )

            perf_df.to_excel(os.path.join(RESULTS_DIR, "performance_table_from_ui.xlsx"), index=False)
            confusion_matrix.to_excel(os.path.join(RESULTS_DIR, "confusion_matrix_from_ui.xlsx"))

            messagebox.showinfo(
                "Success",
                "Dataset analysis completed.\nResults were also saved in the results folder."
            )

        except Exception as e:
            messagebox.showerror("Error", f"Dataset analysis failed.\n\n{e}")
            print(traceback.format_exc())


if __name__ == "__main__":
    root = tk.Tk()
    app = SpeechApp(root)
    root.mainloop()