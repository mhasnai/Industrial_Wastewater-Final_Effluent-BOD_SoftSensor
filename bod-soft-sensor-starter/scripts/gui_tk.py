# -*- coding: utf-8 -*-
"""
Tkinter GUI for wastewater aeration simulation.

Small fixes made:
- Define `full_results = None` before use.
- Ensure DataFrame columns match slider labels so sklearn models get named columns.
- Keep Keras models working with numpy arrays.
- Added __name__ == "__main__" guard.
"""
import tkinter as tk
from tkinter import ttk, filedialog
import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from tensorflow.keras.models import load_model
import pickle
import pandas as pd
import itertools
import threading
import os
import joblib
import warnings
warnings.filterwarnings("ignore", message=".*does not have valid feature names.*")

# Global state
fig_contour = None
cancel_generation = False
cancel_check = False
full_results = None  # NEW: initialize before use

# Map slider feature labels to model training feature names (1:1 mapping here)
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

# Slider definitions: (label, min, max, step, default)
slider_labels = [
    ("F-COD (mg/L)", 450, 600, 50, 500),
    ("F-SS (mg/L)", 450, 900, 0.01, 600),
    ("Flow-Recycle (L/s)", 0, 300, 0.01, 160),
    ("F-TKN (mg/L)", 50, 500, 0.1, 60),
    ("F-NO3 (mg/L)", 50, 1000, 0.1, 300),
    ("SRT (days)", 1, 20, 0.1, 3),
    ("Flow (L/s)", 0, 300, 0.1, 100),
    ("Flow-Aeration (L/s)", 0, 20, 10, 6),
    ("DO (mg/L)", 0, 5, 0.5, 1.5),
    ("F-PH", 0, 10, 1, 4),
    ("Temperature (C)", 10, 40, 1, 28),
]
textbox_labels = ["F-BOD (mg/L)"]

# Main app
def main():
    global root, notebook, main_frame, plot_frame, combination_frame
    global model_selector_var, slider_vars, slider_value_labels, textboxes
    global feature_names, feature_vars, target_var, target_menu, status_label
    global multi_output_model, model_type

    # Create main application window
    root = tk.Tk()
    root.title("Enhanced Wastewater Aeration Process Simulation")
    root.geometry("1200x900")
    root.configure(bg="#f0f0f0")

    # Notebook
    notebook = ttk.Notebook(root)
    notebook.pack(padx=10, pady=10, expand=True)

    # Main tab
    main_frame = ttk.Frame(notebook)
    notebook.add(main_frame, text="Main Interface")

    # Model loader
    def load_selected_model():
        model_path = filedialog.askopenfilename(filetypes=[("Model Files", "*.keras *.h5 *.pkl *.joblib")])
        if model_path:
            model_type_display = (
                "Keras Model" if model_path.endswith(".keras")
                else "H5 Model" if model_path.endswith(".h5")
                else "PKL Model" if model_path.endswith(".pkl")
                else "Joblib Model" if model_path.endswith(".joblib")
                else "Unknown Model"
            )
            model_selector_var.set(model_type_display)

            nonlocal multi_output_model, model_type, target_menu, target_var
            try:
                if model_path.endswith(".keras") or model_path.endswith(".h5"):
                    multi_output_model = load_model(model_path, compile=False)
                    model_type = "keras"
                elif model_path.endswith(".pkl"):
                    with open(model_path, 'rb') as f:
                        multi_output_model = pickle.load(f)
                    model_type = "pkl"
                elif model_path.endswith(".joblib"):
                    multi_output_model = joblib.load(model_path)
                    model_type = "joblib"
                else:
                    print("Unsupported file type.")
                    return

                # Recreate target menu
                target_var = tk.StringVar()
                target_menu = ttk.OptionMenu(plot_frame, target_var, "", *textbox_labels)
                target_menu.grid(row=len(feature_names) + 1, column=0, padx=10, pady=10)
                update_target_menu(textbox_labels)

                predict_values()  # update immediately

            except Exception as e:
                print(f"Error loading model: {e}")
                model_selector_var.set("Select Model")

    model_selector_var = tk.StringVar(value="Select Model")
    model_selector_button = ttk.Button(main_frame, text="Browse Model", command=load_selected_model)
    model_selector_button.grid(row=0, column=0, padx=10, pady=10, sticky="w")
    tk.Label(main_frame, textvariable=model_selector_var, bg="#f0f0f0", font=("Arial", 10)).grid(row=0, column=1, padx=10, pady=10, sticky="w")

    # Sliders
    slider_vars = []
    slider_value_labels = []
    for i, (label, min_val, max_val, step, default) in enumerate(slider_labels):
        tk.Label(main_frame, text=label, bg="#f0f0f0", font=("Arial", 10)).grid(row=i+1, column=0, padx=10, pady=5, sticky="w")
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

    # Predicted textboxes
    textboxes = []
    for i, label in enumerate(textbox_labels):
        tk.Label(main_frame, text=label, bg="#f0f0f0", font=("Arial", 10), anchor="center").grid(
            row=len(slider_labels)+2+(i//6), column=(i%6), padx=5, pady=5, sticky="we"
        )
        text = ttk.Entry(main_frame, font=("Arial", 10), width=15, justify="center")
        text.grid(row=len(slider_labels)+3+(i//6), column=(i%6), padx=5, pady=5, sticky="we")
        textboxes.append(text)

    # Prediction
    def predict_values():
        if not model_selector_var.get().startswith("Select"):
            values = [var.get() for var in slider_vars]
            values_array = np.array([values])
            try:
                if model_type in ["h5", "keras"]:
                    predictions = multi_output_model.predict(values_array)[0]
                elif model_type in ["pkl", "joblib"]:
                    # Name columns in the expected order
                    colnames = [lbl for (lbl, *_rest) in slider_labels]
                    input_df = pd.DataFrame(values_array, columns=colnames)
                    predictions = multi_output_model.predict(input_df)
                predictions = np.ravel(predictions)
                for i, prediction in enumerate(predictions):
                    textboxes[i].delete(0, tk.END)
                    textboxes[i].insert(0, f"{float(prediction):.2f}")
            except Exception as e:
                print(f"Error during prediction: {e}")

    def update_label(var, label):
        label.config(text=f"{var.get():.1f}")
        predict_values()

    predict_values()

    # 3D / heatmap tab
    plot_frame = ttk.Frame(notebook)
    notebook.add(plot_frame, text="3D Surface Plot")

    # Feature selection
    feature_names = [label[0] for label in slider_labels]
    feature_vars = {name: tk.BooleanVar() for name in feature_names}
    for idx, name in enumerate(feature_names):
        tk.Checkbutton(plot_frame, text=name, variable=feature_vars[name]).grid(row=idx, column=0, sticky="w", padx=5, pady=2)

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

    # Save helpers use outer-scope X,Y,Z,fig variables; define placeholders
    X = Y = Z = None
    fig = None

    def save_3d_plot():
        nonlocal fig, X, Y, Z
        file_path = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG files", "*.png"), ("All files", "*.*")])
        if file_path and fig is not None:
            fig.savefig(file_path, dpi=600, transparent=True)
            data_file_path = file_path.replace(".png", "_data.csv")
            data = {"X": X.ravel(), "Y": Y.ravel(), "Z": Z.ravel()}
            pd.DataFrame(data).to_csv(data_file_path, index=False)

    def save_contour_plot():
        global fig_contour
        if fig_contour is None:
            print("No contour plot available to save.")
            return
        file_path = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG files", "*.png"), ("All files", "*.*")])
        if file_path:
            fig_contour.savefig(file_path, dpi=600, transparent=True, bbox_inches='tight', pad_inches=0.1)
            data_file_path = file_path.replace(".png", "_data.csv")
            data = {"X": X.ravel(), "Y": Y.ravel(), "Z": Z.ravel()}
            pd.DataFrame(data).to_csv(data_file_path, index=False)

    def plot_3d_surface():
        nonlocal X, Y, Z, fig, target_var, target_menu
        if model_selector_var.get().startswith("Select"):
            print("Please select a model before plotting.")
            root.after(0, lambda: status_label.config(text="Please select a model before plotting."))
            return

        selected_features = [name for name, var in feature_vars.items() if var.get()]
        if len(selected_features) != 2:
            root.after(0, lambda: status_label.config(text="Please select exactly two features for plotting."))
            return

        target = target_var.get() or textbox_labels[0]

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

        # Name columns for sklearn paths
        colnames = [lbl for (lbl, *_rest) in slider_labels]
        input_df = pd.DataFrame(batch_input, columns=colnames)

        try:
            if model_type in ["h5", "keras"]:
                predictions = multi_output_model.predict(input_df.values)
            else:
                predictions = multi_output_model.predict(input_df)

            if predictions.ndim == 1:
                predictions = np.expand_dims(predictions, axis=1)

            Z = predictions[:, target_idx].reshape(grid_shape)

            fig = Figure(figsize=(8, 6), dpi=100)
            ax = fig.add_subplot(111, projection="3d")
            surf = ax.plot_surface(X, Y, Z, cmap="viridis")

            ax.set_xlabel(selected_features[0], fontsize=12, fontweight='bold')
            ax.set_ylabel(selected_features[1], fontsize=12, fontweight='bold')
            ax.set_zlabel(target, fontsize=12, fontweight='bold')

            fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10, pad=0.1)

            canvas = FigureCanvasTkAgg(fig, plot_frame)
            canvas.get_tk_widget().grid(row=0, column=1, rowspan=20, padx=10, pady=10, sticky="nsew")
            canvas.draw()

        except Exception as e:
            print(f"Error during plotting: {e}")

        # Re-add checkboxes and buttons
        for idx, name in enumerate(feature_names):
            tk.Checkbutton(plot_frame, text=name, variable=feature_vars[name]).grid(row=idx, column=0, sticky="w", padx=5, pady=2)

        ttk.Button(plot_frame, text="Plot 3D Surface", command=plot_3d_surface).grid(row=len(feature_names)+2, column=0, pady=10, sticky="w")
        ttk.Button(plot_frame, text="Plot Heatmap/Contour", command=plot_2d_heatmap_or_contour).grid(row=len(feature_names)+3, column=0, pady=10, sticky="w")
        ttk.Button(plot_frame, text="Save 3D Plot", command=save_3d_plot).grid(row=len(feature_names)+4, column=0, padx=10, pady=10, sticky="w")

    def plot_2d_heatmap_or_contour():
        nonlocal target_var, target_menu, X, Y, Z
        global fig_contour
        if model_selector_var.get().startswith("Select"):
            root.after(0, lambda: status_label.config(text="Please select a model before plotting."))
            return

        selected_features = [name for name, var in feature_vars.items() if var.get()]
        if len(selected_features) != 2:
            root.after(0, lambda: status_label.config(text="Please select exactly two features for plotting."))
            return

        target = target_var.get() or textbox_labels[0]

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

        colnames = [lbl for (lbl, *_rest) in slider_labels]
        input_df = pd.DataFrame(batch_input, columns=colnames)

        try:
            if model_type in ["h5", "keras"]:
                predictions = multi_output_model.predict(input_df.values)
            else:
                predictions = multi_output_model.predict(input_df)

            if predictions.ndim == 1:
                predictions = np.expand_dims(predictions, axis=1)

            Z = predictions[:, target_idx].reshape(grid_shape)

            fig_contour = Figure(figsize=(8, 6), dpi=100)
            ax = fig_contour.add_subplot(111)

            heatmap = ax.imshow(
                Z, extent=[x_vals.min(), x_vals.max(), y_vals.min(), y_vals.max()],
                origin="lower", aspect="auto", cmap="viridis"
            )
            contour = ax.contour(X, Y, Z, colors="black", linewidths=0.8)
            ax.clabel(contour, inline=True, fmt='%d', fontsize=12)

            ax.set_xlabel(selected_features[0], fontsize=12, fontweight='bold')
            ax.set_ylabel(selected_features[1], fontsize=12, fontweight='bold')

            cbar = fig_contour.colorbar(heatmap, ax=ax, label=target)
            cbar.set_label(target, fontsize=12, fontweight='bold')

            canvas = FigureCanvasTkAgg(fig_contour, plot_frame)
            canvas.get_tk_widget().grid(row=0, column=1, rowspan=20, padx=10, pady=10, sticky="nsew")
            canvas.draw()

        except Exception as e:
            print(f"Error during plotting: {e}")

        for idx, name in enumerate(feature_names):
            tk.Checkbutton(plot_frame, text=name, variable=feature_vars[name]).grid(row=idx, column=0, sticky="w", padx=5, pady=2)

        ttk.Button(plot_frame, text="Plot 3D Surface", command=plot_3d_surface).grid(row=len(feature_names)+2, column=0, pady=10, sticky="w")
        ttk.Button(plot_frame, text="Plot Heatmap/Contour", command=plot_2d_heatmap_or_contour).grid(row=len(feature_names)+3, column=0, pady=10, sticky="w")
        ttk.Button(plot_frame, text="Save Contour Plot", command=save_contour_plot).grid(row=len(feature_names)+4, column=0, padx=10, pady=10, sticky="w")

    # Combinations tab
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

            ranges = [np.arange(label[1], label[2] + label[3], label[3]) for label in slider_labels]
            for values in itertools.product(*ranges):
                if cancel_generation:
                    root.after(0, lambda: status_label.config(text="Combination generation canceled."))
                    return
                combinations.append(list(values))
                if len(combinations) >= 1000:
                    df = pd.DataFrame(combinations, columns=[label[0] for label in slider_labels])
                    df.to_csv(file_path, mode='a', header=not os.path.exists(file_path), index=False)
                    combinations = []

            if combinations:
                df = pd.DataFrame(combinations, columns=[label[0] for label in slider_labels])
                df.to_csv(file_path, mode='a', header=not os.path.exists(file_path), index=False)

            root.after(0, lambda: status_label.config(text="Combinations generation completed."))

        threading.Thread(target=worker, daemon=True).start()

    def check_combinations():
        global cancel_check, full_results
        cancel_check = False

        def worker():
            root.after(0, lambda: status_label.config(text="Checking combinations, please wait..."))
            if model_selector_var.get().startswith("Select"):
                root.after(0, lambda: status_label.config(text="No model selected."))
                return

            try:
                file_path = 'feature_combinations.csv'
                if not os.path.exists(file_path):
                    root.after(0, lambda: status_label.config(text="Feature combinations file not found."))
                    return

                df = pd.read_csv(file_path)

                batch_size = 100000
                valid_combinations = []

                # Example target condition (edit as needed)
                target_conditions = [lambda preds: preds[0] > 500]
                final_condition = lambda preds: all(cond(preds) for cond in target_conditions)

                for start in range(0, len(df), batch_size):
                    if cancel_check:
                        root.after(0, lambda: status_label.config(text="Combination check canceled."))
                        return

                    batch = df.iloc[start:start+batch_size]

                    colnames = [lbl for (lbl, *_rest) in slider_labels]
                    batch_named = batch.copy()
                    batch_named.columns = colnames

                    if model_type in ["pkl", "joblib"]:
                        predictions = multi_output_model.predict(batch_named)
                    else:
                        predictions = multi_output_model.predict(batch_named.values)

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
                    root.after(0, lambda: status_label.config(text="No valid combinations found."))

            except Exception as e:
                print(f"Error during combination check: {e}")
                root.after(0, lambda: status_label.config(text="Error occurred during combination check."))

        threading.Thread(target=worker, daemon=True).start()

    def display_results(df_valid, columns):
        global full_results
        displayed_data = df_valid.head(20)
        full_results = df_valid

        for widget in combination_frame.winfo_children():
            widget.grid_forget()

        table = ttk.Treeview(combination_frame, columns=columns, show="headings", height=15)
        for col in columns: table.heading(col, text=col); table.column(col, width=100, anchor="center")
        table.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")

        scrollbar = ttk.Scrollbar(combination_frame, orient="vertical", command=table.yview)
        table.configure(yscrollcommand=scrollbar.set)
        scrollbar.grid(row=0, column=1, sticky="ns")

        for _, row in displayed_data.iterrows():
            table.insert("", "end", values=list(row))

        ttk.Button(combination_frame, text="Save Results", command=save_results).grid(row=1, column=0, padx=10, pady=10, sticky="w")

    def save_results():
        if full_results is None or full_results.empty:
            print("No data to save.")
            return
        file_path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV files", "*.csv"), ("All files", "*.*")])
        if file_path:
            full_results.to_csv(file_path, index=False)

    def initialize_combination_tab():
        ttk.Button(combination_frame, text="Generate Combinations", command=generate_combinations).grid(row=0, column=0, padx=10, pady=10, sticky="w")
        ttk.Button(combination_frame, text="Check Combinations", command=check_combinations).grid(row=1, column=0, padx=10, pady=10, sticky="w")
        ttk.Button(combination_frame, text="Cancel Generating/Checking Combinations", command=lambda: cancel_process()).grid(row=2, column=0, padx=10, pady=10, sticky="w")

        global status_label
        status_label = tk.Label(combination_frame, text="", bg="#f0f0f0", font=("Arial", 10), anchor="w")
        status_label.grid(row=3, column=0, padx=10, pady=10, sticky="w")

    def cancel_process():
        global cancel_generation, cancel_check
        cancel_generation = True
        cancel_check = True
        if status_label.winfo_exists():
            status_label.config(text="Operation canceled.")

    def reset_all():
        global cancel_generation, cancel_check, target_menu, target_var
        cancel_generation = True
        cancel_check = True
        for i, (_, _, _, _, default) in enumerate(slider_labels):
            slider_vars[i].set(default)
        predict_values()
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
        for widget in combination_frame.winfo_children():
            widget.destroy()
        initialize_combination_tab()
        print("All functionality has been reset.")

    ttk.Button(plot_frame, text="Plot 3D Surface", command=plot_3d_surface).grid(row=len(feature_names) + 2, column=0, pady=10, sticky="w")
    ttk.Button(plot_frame, text="Plot Heatmap/Contour", command=plot_2d_heatmap_or_contour).grid(row=len(feature_names) + 3, column=0, pady=10, sticky="w")

    # Combinations tab initial controls
    ttk.Button(combination_frame, text="Generate Combinations", command=generate_combinations).grid(row=0, column=0, padx=10, pady=10, sticky="w")
    ttk.Button(combination_frame, text="Check Combinations", command=check_combinations).grid(row=1, column=0, padx=10, pady=10, sticky="w")
    ttk.Button(combination_frame, text="Cancel Generating/Checking Combinations", command=lambda: cancel_process()).grid(row=2, column=0, padx=10, pady=10, sticky="w")
    status_label = tk.Label(combination_frame, text="", bg="#f0f0f0", font=("Arial", 10), anchor="w")
    status_label.grid(row=3, column=0, padx=10, pady=10, sticky="w")

    # Reset button
    ttk.Button(main_frame, text="Reset", command=reset_all).grid(row=0, column=6, padx=10, pady=10, sticky="e")

    root.mainloop()

if __name__ == "__main__":
    main()
