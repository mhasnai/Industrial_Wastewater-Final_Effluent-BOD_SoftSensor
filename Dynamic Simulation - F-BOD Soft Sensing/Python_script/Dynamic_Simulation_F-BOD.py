#!/usr/bin/env python
# coding: utf-8

# In[13]:


import tkinter as tk
from tkinter import ttk, filedialog
import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
# NOTE: Do NOT import tensorflow/keras at top-level (we lazy-load later)
import pickle
import pandas as pd
import itertools
import threading
import os
import joblib
import warnings

warnings.filterwarnings("ignore", message=".*does not have valid feature names.*")

# -------------------------
# Global state
# -------------------------
fig_contour = None            # For the contour plot figure
cancel_generation = False
cancel_check = False
full_results = None           # For saving combination results
multi_output_model = None
model_type = None

# -------------------------
# Feature mapping (GUI label -> model feature name)
# -------------------------
feature_name_mapping = {
    "F-COD (mg/L)": "F-COD (mg/L)",
    "F-SS (mg/L)": "F-SS (mg/L)",
    "Flow-Recycle (L/s)": "Flow-Recycle (L/s)",
    "F-TKN (mg/L)": "F-TKN (mg/L)",
    "F-NO3 (mg/L)": "F-NO3 (mg/L)",
    "SRT (days)": "SRT (days)",
    "Flow (L/s)": "Flow (L/s)",
    "Flow-Aeration (L/s)": "Flow-Aeration (L/s)",
    "DO (mg/L)": "DO (mg/L)",
    "F-PH": "F-PH",
    "Temperature (C)": "Temperature (C)",
}

# -------------------------
# Main window
# -------------------------
root = tk.Tk()
root.title("Enhanced Industrial Wastewater F-BOD Simulation")
root.geometry("1200x900")
root.configure(bg="#f0f0f0")

notebook = ttk.Notebook(root)
notebook.pack(padx=10, pady=10, expand=True)

main_frame = ttk.Frame(notebook)
notebook.add(main_frame, text="Main Interface")

# -------------------------
# Model loading (lazy import for Keras/TensorFlow)
# -------------------------
def load_selected_model():
    """
    Browse for and load a model file.
    Supports: .keras / .h5 (tf.keras / Keras 3), .pkl, .joblib
    Lazy-loads TensorFlow/Keras to avoid DLL issues until actually needed.
    """
    from tkinter import messagebox  # local import so this function is drop-in

    model_path = filedialog.askopenfilename(
        title="Select model file",
        filetypes=[
            ("Model Files", "*.keras *.h5 *.pkl *.joblib"),
            ("Keras / H5", "*.keras *.h5"),
            ("Pickle", "*.pkl"),
            ("Joblib", "*.joblib"),
            ("All files", "*.*"),
        ]
    )
    if not model_path:
        return

    global multi_output_model, model_type, target_menu, target_var

    # small helper to show errors both in console and UI
    def _show_error(msg: str):
        print(msg)
        try:
            status_label.config(text=msg)
        except Exception:
            pass
        try:
            messagebox.showerror("Model load error", msg)
        except Exception:
            pass

    try:
        # -------- KERAS / H5 --------
        if model_path.lower().endswith((".keras", ".h5")):
            # Try TensorFlow's tf.keras first (your TF=2.19 should handle most .keras/.h5)
            try:
                from tensorflow.keras.models import load_model as tf_load_model
                multi_output_model = tf_load_model(model_path, compile=False)
                model_type = "keras"  # tf.keras
                model_selector_var.set("Keras Model")
                status_label.config(text=f"Loaded Keras model: {model_path}")
            except Exception as tf_err:
                print(f"tf.keras load failed, trying Keras 3 fallback: {tf_err}")
                # Fallback to standalone Keras (Keras 3)
                try:
                    from keras.models import load_model as k3_load_model
                    multi_output_model = k3_load_model(model_path, compile=False)
                    model_type = "keras3"
                    model_selector_var.set("Keras 3 Model")
                    status_label.config(text=f"Loaded Keras 3 model: {model_path}")
                except Exception as k3_err:
                    _show_error(
                        "Failed to load .keras/.h5 model with both tf.keras and Keras 3.\n\n"
                        f"tf.keras error: {tf_err}\n\nKeras 3 error: {k3_err}"
                    )
                    model_selector_var.set("Select Model")
                    return

        # -------- PICKLE --------
        elif model_path.lower().endswith(".pkl"):
            try:
                import pickle as _pkl
                with open(model_path, "rb") as f:
                    multi_output_model = _pkl.load(f)
                model_type = "pkl"
                model_selector_var.set("PKL Model")
                status_label.config(text=f"Loaded PKL model: {model_path}")
            except Exception as e:
                _show_error(f"Failed to load .pkl model:\n{e}")
                model_selector_var.set("Select Model")
                return

        # -------- JOBLIB --------
        elif model_path.lower().endswith(".joblib"):
            try:
                import joblib as _joblib
                multi_output_model = _joblib.load(model_path)
                model_type = "joblib"
                model_selector_var.set("Joblib Model")
                status_label.config(text=f"Loaded Joblib model: {model_path}")
            except Exception as e:
                _show_error(f"Failed to load .joblib model:\n{e}")
                model_selector_var.set("Select Model")
                return

        # -------- UNKNOWN --------
        else:
            _show_error("Unsupported file type. Please choose .keras, .h5, .pkl, or .joblib.")
            model_selector_var.set("Select Model")
            return

        # Recreate / refresh target dropdown (plot tab might have cleared it earlier)
        try:
            target_var = tk.StringVar()
            target_menu = ttk.OptionMenu(plot_frame, target_var, "", *textbox_labels)
            target_menu.grid(row=len(feature_names) + 1, column=0, padx=10, pady=10)
            update_target_menu(textbox_labels)
        except Exception as e:
            # Not fatal for predictions—continue
            print(f"Warning: could not refresh target menu: {e}")

        # Kick off an immediate prediction to populate textboxes
        try:
            predict_values()
        except Exception as e:
            _show_error(f"Model loaded, but prediction failed:\n{e}")

    except Exception as e:
        _show_error(f"Unexpected error while loading model:\n{e}")
        model_selector_var.set("Select Model")


# -------------------------
# Model selector UI
# -------------------------
model_selector_var = tk.StringVar(value="Select Model")
model_selector_button = ttk.Button(main_frame, text="Browse Model", command=load_selected_model)
model_selector_button.grid(row=0, column=0, padx=10, pady=10, sticky="w")
tk.Label(main_frame, textvariable=model_selector_var, bg="#f0f0f0", font=("Arial", 10)).grid(
    row=0, column=1, padx=10, pady=10, sticky="w"
)

# -------------------------
# Sliders
# -------------------------
slider_labels = [
    ("F-COD (mg/L)",        450, 600, 50,   500),
    ("F-SS (mg/L)",         450, 900, 0.01, 600),
    ("Flow-Recycle (L/s)",  0,   300, 0.01, 160),
    ("F-TKN (mg/L)",        50,  500, 0.1,  60),
    ("F-NO3 (mg/L)",        50,  1000,0.1,  300),
    ("SRT (days)",          1,   20,  0.1,  3),
    ("Flow (L/s)",          0,   300, 0.1,  100),
    ("Flow-Aeration (L/s)", 0,   20,  10,   6),
    ("DO (mg/L)",           0,   5,   0.5,  1.5),
    ("F-PH",                0,   10,  1,    4),
    ("Temperature (C)",     10,  40,  1,    28),
]

slider_vars = []
slider_value_labels = []

def update_label(var, label):
    label.config(text=f"{var.get():.1f}")
    predict_values()

for i, (label, min_val, max_val, step, default) in enumerate(slider_labels):
    tk.Label(main_frame, text=label, bg="#f0f0f0", font=("Arial", 10)).grid(
        row=i+1, column=0, padx=10, pady=5, sticky="w"
    )
    slider_var = tk.DoubleVar(value=default)
    slider = ttk.Scale(
        main_frame, from_=min_val, to=max_val, orient="horizontal",
        variable=slider_var, length=300,
        command=lambda e, var=slider_var, lbl=i: update_label(var, slider_value_labels[lbl])
    )
    slider.grid(row=i+1, column=1, columnspan=4, padx=10, pady=5, sticky="we")
    value_label = tk.Label(main_frame, text=f"{default}", bg="#f0f0f0", font=("Arial", 10))
    value_label.grid(row=i+1, column=5, padx=10, pady=5)
    slider_value_labels.append(value_label)
    slider_vars.append(slider_var)

# -------------------------
# Predicted features (textbox outputs)
# -------------------------
textbox_labels = ["F-BOD (mg/L)"]
textboxes = []

for i, label in enumerate(textbox_labels):
    tk.Label(main_frame, text=label, bg="#f0f0f0", font=("Arial", 10), anchor="center").grid(
        row=len(slider_labels) + 2 + (i // 6), column=(i % 6), padx=5, pady=5, sticky="we"
    )
    text = ttk.Entry(main_frame, font=("Arial", 10), width=15, justify="center")
    text.grid(row=len(slider_labels) + 3 + (i // 6), column=(i % 6), padx=5, pady=5, sticky="we")
    textboxes.append(text)

def predict_values():
    """Predict textbox values whenever sliders move or a model loads."""
    if model_selector_var.get().startswith("Select") or multi_output_model is None:
        return
    values = [var.get() for var in slider_vars]
    values_array = np.array([values])
    try:
        if model_type in ["h5", "keras", "keras3"]:
            predictions = multi_output_model.predict(values_array)[0]
        elif model_type in ["pkl", "joblib"]:
            input_df = pd.DataFrame(values_array)
            predictions = multi_output_model.predict(input_df)
        else:
            return

        predictions = np.ravel(predictions)
        for i, prediction in enumerate(predictions[:len(textboxes)]):
            textboxes[i].delete(0, tk.END)
            textboxes[i].insert(0, f"{float(prediction):.2f}")
    except Exception as e:
        print(f"Error during prediction: {e}")

# Initial prediction
predict_values()

# -------------------------
# Plot tab (3D + Heatmap/Contour)
# -------------------------
plot_frame = ttk.Frame(notebook)
notebook.add(plot_frame, text="3D Surface Plot")

feature_names = [label[0] for label in slider_labels]
feature_vars = {name: tk.BooleanVar() for name in feature_names}

for idx, name in enumerate(feature_names):
    tk.Checkbutton(plot_frame, text=name, variable=feature_vars[name]).grid(
        row=idx, column=0, sticky="w", padx=5, pady=2
    )

def update_target_menu(labels):
    target_menu['menu'].delete(0, 'end')
    for label in labels:
        target_menu['menu'].add_command(label=label, command=lambda value=label: target_var.set(value))
    if labels:
        target_var.set(labels[0])

target_var = tk.StringVar()
target_menu = ttk.OptionMenu(plot_frame, target_var, "", *textbox_labels)
target_menu.grid(row=len(feature_names) + 1, column=0, padx=10, pady=10)
update_target_menu(textbox_labels)

def save_3d_plot():
    file_path = filedialog.asksaveasfilename(defaultextension=".png",
                                             filetypes=[("PNG files", "*.png"), ("All files", "*.*")])
    if file_path:
        fig.savefig(file_path, dpi=600, transparent=True)
        print(f"3D plot saved to {file_path}")
        data_file_path = file_path.replace(".png", "_data.csv")
        pd.DataFrame({"X": X.ravel(), "Y": Y.ravel(), "Z": Z.ravel()}).to_csv(data_file_path, index=False)
        print(f"3D plot data saved to {data_file_path}")

def save_contour_plot():
    global fig_contour
    if fig_contour is None:
        print("No contour plot available to save.")
        return
    file_path = filedialog.asksaveasfilename(defaultextension=".png",
                                             filetypes=[("PNG files", "*.png"), ("All files", "*.*")])
    if file_path:
        fig_contour.savefig(file_path, dpi=600, transparent=True, bbox_inches='tight', pad_inches=0.1)
        print(f"Contour plot saved to {file_path}")
        data_file_path = file_path.replace(".png", "_data.csv")
        pd.DataFrame({"X": X.ravel(), "Y": Y.ravel(), "Z": Z.ravel()}).to_csv(data_file_path, index=False)
        print(f"Contour plot data saved to {data_file_path}")

def plot_3d_surface():
    global target_menu, target_var, X, Y, Z, fig

    if model_selector_var.get().startswith("Select") or multi_output_model is None:
        print("Please select a model before plotting.")
        root.after(0, lambda: status_label.config(text="Please select a model before plotting."))
        return

    selected_features = [name for name, var in feature_vars.items() if var.get()]
    if len(selected_features) != 2:
        root.after(0, lambda: status_label.config(text="Please select exactly two features for plotting."))
        return

    target = target_var.get()
    if not target:
        root.after(0, lambda: status_label.config(text="Please select a target feature for plotting."))
        return

    # Clear plot area
    for widget in plot_frame.winfo_children():
        widget.destroy()

    # Recreate target menu
    target_var = tk.StringVar()
    target_menu = ttk.OptionMenu(plot_frame, target_var, "", *textbox_labels)
    target_menu.grid(row=len(feature_names) + 1, column=0, padx=10, pady=10)
    target_var.set(textbox_labels[0])

    x_idx = feature_names.index(selected_features[0])
    y_idx = feature_names.index(selected_features[1])
    target_idx = textbox_labels.index(target)

    x_vals = np.linspace(slider_labels[x_idx][1], slider_labels[x_idx][2], 50)
    y_vals = np.linspace(slider_labels[y_idx][1], slider_labels[y_idx][2], 50)
    X, Y = np.meshgrid(x_vals, y_vals)

    grid_shape = X.shape
    batch_input = np.zeros((grid_shape[0] * grid_shape[1], len(slider_vars)))
    for i, slider in enumerate(slider_vars):
        batch_input[:, i] = slider.get()
    batch_input[:, x_idx] = X.ravel()
    batch_input[:, y_idx] = Y.ravel()

    input_df = pd.DataFrame(batch_input)
    input_df.columns = [feature_name_mapping.get(col, col) for col in input_df.columns]

    try:
        if model_type in ["h5", "keras", "keras3"]:
            predictions = multi_output_model.predict(input_df.values)
        elif model_type in ["pkl", "joblib"]:
            predictions = multi_output_model.predict(input_df)
        if len(predictions.shape) == 1:
            predictions = np.expand_dims(predictions, axis=1)
        Z = predictions[:, target_idx].reshape(grid_shape)

        fig = Figure(figsize=(8, 6), dpi=100)
        ax = fig.add_subplot(111, projection="3d")
        surf = ax.plot_surface(X, Y, Z, cmap="viridis")
        ax.set_xlabel(selected_features[0], fontsize=14, fontweight='bold', fontname='Arial')
        ax.set_ylabel(selected_features[1], fontsize=14, fontweight='bold', fontname='Arial')
        ax.set_zlabel(target, fontsize=14, fontweight='bold', fontname='Arial')
        fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10, pad=0.1)

        canvas = FigureCanvasTkAgg(fig, plot_frame)
        canvas.get_tk_widget().grid(row=0, column=1, rowspan=20, padx=10, pady=10, sticky="nsew")
        canvas.draw()
    except Exception as e:
        print(f"Error during plotting: {e}")

    # Re-add controls
    for idx, name in enumerate(feature_names):
        tk.Checkbutton(plot_frame, text=name, variable=feature_vars[name]).grid(row=idx, column=0, sticky="w", padx=5, pady=2)
    ttk.Button(plot_frame, text="Plot 3D Surface", command=plot_3d_surface).grid(row=len(feature_names) + 2, column=0, pady=10, sticky="w")
    ttk.Button(plot_frame, text="Plot Heatmap/Contour", command=plot_2d_heatmap_or_contour).grid(row=len(feature_names) + 3, column=0, pady=10, sticky="w")
    ttk.Button(plot_frame, text="Save 3D Plot", command=save_3d_plot).grid(row=len(feature_names) + 4, column=0, padx=10, pady=10, sticky="w")

ttk.Button(plot_frame, text="Plot 3D Surface", command=plot_3d_surface).grid(row=len(feature_names) + 2, column=0, pady=10, sticky="w")

def plot_2d_heatmap_or_contour():
    global target_menu, target_var, fig_contour, X, Y, Z

    if model_selector_var.get().startswith("Select") or multi_output_model is None:
        root.after(0, lambda: status_label.config(text="Please select a model before plotting."))
        return

    selected_features = [name for name, var in feature_vars.items() if var.get()]
    if len(selected_features) != 2:
        root.after(0, lambda: status_label.config(text="Please select exactly two features for plotting."))
        return

    target = target_var.get()
    if not target:
        root.after(0, lambda: status_label.config(text="Please select a target feature for plotting."))
        return

    # Clear plot area
    for widget in plot_frame.winfo_children():
        widget.destroy()

    target_var = tk.StringVar()
    target_menu = ttk.OptionMenu(plot_frame, target_var, "", *textbox_labels)
    target_menu.grid(row=len(feature_names) + 1, column=0, padx=10, pady=10)
    target_var.set(textbox_labels[0])

    x_idx = feature_names.index(selected_features[0])
    y_idx = feature_names.index(selected_features[1])
    target_idx = textbox_labels.index(target)

    x_vals = np.linspace(slider_labels[x_idx][1], slider_labels[x_idx][2], 50)
    y_vals = np.linspace(slider_labels[y_idx][1], slider_labels[y_idx][2], 50)
    X, Y = np.meshgrid(x_vals, y_vals)

    grid_shape = X.shape
    batch_input = np.zeros((grid_shape[0] * grid_shape[1], len(slider_vars)))
    for i, slider in enumerate(slider_vars):
        batch_input[:, i] = slider.get()
    batch_input[:, x_idx] = X.ravel()
    batch_input[:, y_idx] = Y.ravel()

    input_df = pd.DataFrame(batch_input)
    input_df.columns = [feature_name_mapping.get(col, col) for col in input_df.columns]

    try:
        if model_type in ["h5", "keras", "keras3"]:
            predictions = multi_output_model.predict(input_df.values)
        elif model_type in ["pkl", "joblib"]:
            predictions = multi_output_model.predict(input_df)
        if len(predictions.shape) == 1:
            predictions = np.expand_dims(predictions, axis=1)

        Z = predictions[:, target_idx].reshape(grid_shape)

        fig_contour = Figure(figsize=(8, 6), dpi=100)
        ax = fig_contour.add_subplot(111)

        heatmap = ax.imshow(
            Z, extent=[x_vals.min(), x_vals.max(), y_vals.min(), y_vals.max()],
            origin="lower", aspect="auto", cmap="viridis"
        )
        contour = ax.contour(X, Y, Z, colors="black", linewidths=0.8)
        contour_labels = ax.clabel(contour, inline=True, fmt='%d', fontsize=14)
        for txt in contour_labels:
            txt.set_fontname("Arial")
            txt.set_fontsize(14)

        ax.set_xlabel(selected_features[0], fontsize=16, fontweight='bold', fontname='Arial')
        ax.set_ylabel(selected_features[1], fontsize=16, fontweight='bold', fontname='Arial')
        ax.tick_params(axis='both', labelsize=14)
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontname("Arial")
            label.set_fontsize(14)

        cbar = fig_contour.colorbar(heatmap, ax=ax, label=target)
        cbar.set_label(target, fontsize=16, fontweight='bold', fontname='Arial')
        cbar.ax.tick_params(labelsize=14)
        for label in cbar.ax.get_yticklabels():
            label.set_fontname("Arial")
            label.set_fontsize(14)

        canvas = FigureCanvasTkAgg(fig_contour, plot_frame)
        canvas.get_tk_widget().grid(row=0, column=1, rowspan=20, padx=10, pady=10, sticky="nsew")
        canvas.draw()

    except Exception as e:
        print(f"Error during plotting: {e}")

    # Re-add controls
    for idx, name in enumerate(feature_names):
        tk.Checkbutton(plot_frame, text=name, variable=feature_vars[name]).grid(row=idx, column=0, sticky="w", padx=5, pady=2)
    ttk.Button(plot_frame, text="Plot 3D Surface", command=plot_3d_surface).grid(row=len(feature_names) + 2, column=0, pady=10, sticky="w")
    ttk.Button(plot_frame, text="Plot Heatmap/Contour", command=plot_2d_heatmap_or_contour).grid(row=len(feature_names) + 3, column=0, pady=10, sticky="w")
    ttk.Button(plot_frame, text="Save Contour Plot", command=save_contour_plot).grid(row=len(feature_names) + 4, column=0, padx=10, pady=10, sticky="w")

ttk.Button(plot_frame, text="Plot Heatmap/Contour", command=plot_2d_heatmap_or_contour).grid(row=len(feature_names) + 3, column=0, pady=10, sticky="w")

# -------------------------
# Feature Combinations tab
# -------------------------
combination_frame = ttk.Frame(notebook)
notebook.add(combination_frame, text="Feature Combinations")

def generate_combinations():
    def worker():
        global cancel_generation
        cancel_generation = False
        root.after(0, lambda: status_label.config(text="Generating combinations, please wait..."))

        combinations = []
        file_path = 'feature_combinations.csv'
        if os.path.exists(file_path):
            os.remove(file_path)

        for values in itertools.product(*[
            np.arange(label[1], label[2] + label[3], label[3]) for label in slider_labels
        ]):
            if cancel_generation:
                root.after(0, lambda: status_label.config(text="Combination generation canceled."))
                return

            combinations.append(list(values))
            if len(combinations) >= 1000:
                df = pd.DataFrame(combinations, columns=[label[0] for label in slider_labels])
                df.columns = [feature_name_mapping.get(col, col) for col in df.columns]
                df.to_csv(file_path, mode='a', header=not os.path.exists(file_path), index=False)
                combinations = []

        if combinations:
            df = pd.DataFrame(combinations, columns=[label[0] for label in slider_labels])
            df.columns = [feature_name_mapping.get(col, col) for col in df.columns]
            df.to_csv(file_path, mode='a', header=not os.path.exists(file_path), index=False)

        root.after(0, lambda: status_label.config(text="Combinations generation completed."))
    threading.Thread(target=worker, daemon=True).start()

def check_combinations():
    global cancel_check
    cancel_check = False

    def worker():
        root.after(0, lambda: status_label.config(text="Checking combinations, please wait...") if status_label.winfo_exists() else None)
        if model_selector_var.get().startswith("Select") or multi_output_model is None:
            root.after(0, lambda: status_label.config(text="No model selected.") if status_label.winfo_exists() else None)
            return
        try:
            file_path = 'feature_combinations.csv'
            if not os.path.exists(file_path):
                root.after(0, lambda: status_label.config(text="Feature combinations file not found.") if status_label.winfo_exists() else None)
                return

            df = pd.read_csv(file_path)
            df.columns = [feature_name_mapping.get(col, col) for col in df.columns]

            batch_size = 100000
            valid_combinations = []

            # Example condition on prediction(s) — adjust as needed
            target_conditions = [
                lambda preds: preds[0] > 500,  # Example: F-BOD > 500
            ]
            final_condition = lambda preds: all(cond(preds) for cond in target_conditions)

            for start in range(0, len(df), batch_size):
                if cancel_check:
                    root.after(0, lambda: status_label.config(text="Combination check canceled.") if status_label.winfo_exists() else None)
                    return

                batch = df.iloc[start:start + batch_size]

                if model_type in ["pkl", "joblib"]:
                    predictions = multi_output_model.predict(batch)
                elif model_type in ["h5", "keras", "keras3"]:
                    predictions = multi_output_model.predict(batch.values)

                for i, prediction in enumerate(predictions):
                    if isinstance(prediction, np.ndarray):
                        prediction = prediction.flatten()
                    if final_condition(prediction):
                        valid_combinations.append(list(batch.iloc[i]) + list(prediction))

            if valid_combinations:
                columns = list(df.columns) + textbox_labels
                df_valid = pd.DataFrame(valid_combinations, columns=columns)
                root.after(0, lambda: display_results(df_valid, columns))
            else:
                root.after(0, lambda: status_label.config(text="No valid combinations found.") if status_label.winfo_exists() else None)

        except Exception as e:
            print(f"Error during combination check: {e}")
            root.after(0, lambda: status_label.config(text="Error occurred during combination check.") if status_label.winfo_exists() else None)

    threading.Thread(target=worker, daemon=True).start()

def display_results(df_valid, columns):
    displayed_data = df_valid.head(20)
    global full_results
    full_results = df_valid

    for widget in combination_frame.winfo_children():
        widget.grid_forget()

    table = ttk.Treeview(combination_frame, columns=columns, show="headings", height=15)
    for col in columns:
        table.heading(col, text=col)
        table.column(col, width=100, anchor="center")
    table.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")

    scrollbar = ttk.Scrollbar(combination_frame, orient="vertical", command=table.yview)
    table.configure(yscrollcommand=scrollbar.set)
    scrollbar.grid(row=0, column=1, sticky="ns")

    for _, row in displayed_data.iterrows():
        table.insert("", "end", values=list(row))

    save_button = ttk.Button(combination_frame, text="Save Results", command=save_results)
    save_button.grid(row=1, column=0, padx=10, pady=10, sticky="w")

def save_results():
    if full_results is None or full_results.empty:
        print("No data to save.")
        return
    file_path = filedialog.asksaveasfilename(defaultextension=".csv",
                                             filetypes=[("CSV files", "*.csv"), ("All files", "*.*")])
    if file_path:
        full_results.to_csv(file_path, index=False)
        print(f"Results saved to {file_path}")

def initialize_combination_tab():
    combination_button = ttk.Button(combination_frame, text="Generate Combinations", command=generate_combinations)
    combination_button.grid(row=0, column=0, padx=10, pady=10, sticky="w")

    check_button = ttk.Button(combination_frame, text="Check Combinations", command=check_combinations)
    check_button.grid(row=1, column=0, padx=10, pady=10, sticky="w")

    cancel_button = ttk.Button(combination_frame, text="Cancel Generating/Checking Combinations",
                               command=lambda: cancel_process("generate"))
    cancel_button.grid(row=2, column=0, padx=10, pady=10, sticky="w")

    global status_label
    status_label = tk.Label(combination_frame, text="", bg="#f0f0f0", font=("Arial", 10), anchor="w")
    status_label.grid(row=3, column=0, padx=10, pady=10, sticky="w")

def cancel_process(process_type):
    global cancel_generation, cancel_check
    cancel_generation = True
    if status_label.winfo_exists():
        status_label.config(text="Combination generation canceled.")
    cancel_check = True
    if status_label.winfo_exists():
        status_label.config(text="Combination checking canceled.")

initialize_combination_tab()

# Reset button on Main tab
def reset_all():
    global cancel_generation, cancel_check, target_menu, target_var
    cancel_generation = True
    cancel_check = True

    for i, (_, _, _, _, default) in enumerate(slider_labels):
        slider_vars[i].set(default)

    predict_values()

    # Reset plot tab
    for widget in plot_frame.winfo_children():
        widget.destroy()

    for idx, name in enumerate(feature_names):
        tk.Checkbutton(plot_frame, text=name, variable=feature_vars[name]).grid(row=idx, column=0, sticky="w", padx=5, pady=2)

    target_var = tk.StringVar()
    target_menu = ttk.OptionMenu(plot_frame, target_var, "", *textbox_labels)
    target_menu.grid(row=len(feature_names) + 1, column=0, padx=10, pady=10)
    target_var.set(textbox_labels[0])

    ttk.Button(plot_frame, text="Plot 3D Surface", command=plot_3d_surface).grid(row=len(feature_names) + 2, column=0, pady=10, sticky="w")
    ttk.Button(plot_frame, text="Plot Heatmap/Contour", command=plot_2d_heatmap_or_contour).grid(row=len(feature_names) + 3, column=0, pady=10, sticky="w")

    # Reset combinations tab
    for widget in combination_frame.winfo_children():
        widget.destroy()
    initialize_combination_tab()

    print("All functionality has been reset.")

reset_button = ttk.Button(main_frame, text="Reset", command=reset_all)
reset_button.grid(row=0, column=6, padx=10, pady=10, sticky="e")

# -------------------------
# Standard Windows-freeze guard
# -------------------------
if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    root.mainloop()


# In[ ]:




