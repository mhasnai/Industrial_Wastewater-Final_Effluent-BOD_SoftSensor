Dynamic Simulation – F-BOD Soft Sensing
========================================

Predictive & simulation tool (GUI) for **soft sensing of final-effluent BOD (F‑BOD)** in industrial wastewater.
The Python app lets you **load a trained model**, tweak input parameters with sliders, and **see the impact on F‑BOD**. It can also generate **3D surfaces** and **2D contour/heatmaps** to explore feature combinations.

This repository contains the Python script, trained Neural Network model, and environment specifications for predicting and simulating final effluent Biological Oxygen Demand (F-BOD) in industrial wastewater treatment, as described in our paper:

    Hassnain, Muhammad; Lee, Sarada M.W.; Azhar, Muhammad Rizwan (2025).
    "Soft Sensing of Biological Oxygen Demand in Industrial Wastewater Using Machine Learning Models".
    Available at SSRN: https://ssrn.com/abstract=5360563
    DOI: http://dx.doi.org/10.2139/ssrn.5360563

License
-------
This code and the provided model are released under the **MIT License**.
You may use, modify, and distribute them with attribution. See the `LICENSE` file.

Repository layout
-----------------
Dynamic Simulation - F-BOD Soft Sensing/
│
├── .gitattributes                           # Git attributes configuration
├── .gitignore                               # Ignored files list for Git
├── LICENSE                                  # MIT License file
├── README.md                                # Markdown version of this documentation
├── README.txt                               # Plain text version of this documentation
├── requirements                             # Optional pip requirements file
│
├── Python_script/
│   └── Dynamic_Simulation_F-BOD.py          # Main GUI Python script for dynamic simulation
│
├── Python_Environment/
│   └── AI_BOD_Softsenor_PyEnv.yaml          # Conda environment specification
│
├── NN_AI-model/
│   └── NNmodel.keras                        # Trained Neural Network model for prediction & simulation
│
└── Sample Dataset/                          # Folder containing sample data for illustration
    └── ActualData_Subsample.xlsx

Detailed procedure
------------------
1) **Install Conda** (Miniconda/Anaconda).
2) **Create & activate** the environment:
       conda env create -f Python_Environment/AI_BOD_Softsenor_PyEnv.yaml
       conda activate AI_BOD_Softsenor_PyEnv
3) **Start the GUI**:
       python Python_script/Dynamic_Simulation_F-BOD.py
4) **Load model**:
   - Click **Browse Model** and choose `NN_AI-model/NNmodel.keras` (or your own `.keras/.h5/.pkl/.joblib` file).
   - The status text updates to confirm successful load.
5) **Live prediction**:
   - Move sliders to change inputs; predicted **F‑BOD (mg/L)** updates in the output textbox.
6) **Explore feature combinations (3D/Contour)**:
   - Click the **3D Surface Plot** tab.
   - Tick **exactly two** feature checkboxes (these define the **combination**).
   - Ensure the **Target** dropdown shows `F‑BOD (mg/L)` (default).
   - Click **Plot 3D Surface** to compute a **50×50 grid** (X,Y swept over their full ranges; all other features are **held at the current slider values**). A rotatable 3D surface is rendered.
   - Click **Plot Heatmap/Contour** to view the same grid as a 2D heatmap with overlaid contour lines.
   - Use **Save 3D Plot** or **Save Contour Plot** to export a high‑resolution PNG. A companion CSV (`*_data.csv`) with the grid values (X, Y, Z=F‑BOD) is saved automatically.
   - To change the combination, tick a different pair of features and re‑plot.
   - Grid resolution (50) can be modified in code where `np.linspace(..., 50)` is used.
7) **Close** the window to exit.

GUI: controls and buttons
-------------------------
**Main Interface tab**
- **Browse Model**: Opens a file chooser to load a model (`.keras`, `.h5`, `.pkl`, `.joblib`). Uses lazy loading; error messages appear in the status text if loading fails.
- **Sliders (11)**: Inputs to the model. Adjust to see real‑time F‑BOD predictions. A numeric label shows the current value.
- **Predicted Output**: Textbox labelled `F‑BOD (mg/L)`; updates after model load and on slider change.
- **Status text**: Short messages about load/prediction issues and guidance.

**3D Surface Plot tab**
- **Feature checkboxes**: Select **two** inputs to define the combination for plotting.
- **Target dropdown**: Choose which output to plot (currently `F‑BOD (mg/L)`; field is present for future multi‑output models).
- **Plot 3D Surface**: Generates a 3D surface (X,Y → predicted F‑BOD) using a 50×50 grid; other inputs fixed at current slider values.
- **Plot Heatmap/Contour**: Generates a 2D heatmap with contour labels for the same grid.
- **Save 3D Plot**: Saves the current 3D figure as PNG and a `*_data.csv` with the underlying grid.
- **Save Contour Plot**: Saves the current contour/heatmap as PNG and a `*_data.csv` with the grid (appears after a contour plot is created).


**Feature Combinations tab**
This tab automates bulk testing of **all possible input value combinations** to find conditions that meet certain model-predicted criteria.

**Buttons & functions:**
- **Generate Combinations** –
  Creates a CSV file (`feature_combinations.csv`) containing every combination of all input features based on their slider ranges & step sizes.
  ⚠ This can be a very large file; generation runs in the background.
  The status label at the bottom reports progress.
- **Check Combinations** –
  Loads the generated CSV, runs the model on each row, and filters to combinations meeting coded conditions (default: *F-BOD > 500*).
  Matching results are displayed in a table (top 20 rows) and stored in memory for export.
- **Save Results** –
  Appears after a successful check; saves the filtered combinations to a chosen `.csv`.
- **Cancel Generating/Checking Combinations** –
  Stops the current process immediately.
- **Status label** –
  Shows progress updates, errors, or cancellation confirmations.

**How it works:**
1. The script iterates through every slider range at its defined increment.
2. Combinations are written in batches to avoid memory overload.
3. Checking uses the loaded model to predict for each combination, applying the criteria in `check_combinations()`.
4. Results can be browsed and exported.

Feature glossary (inputs → model)
---------------------------------
• **F‑COD (mg/L)** — Final‑effluent *Chemical Oxygen Demand*  
• **F‑SS (mg/L)** — Final‑effluent *Suspended Solids*  
• **Flow‑Recycle (L/s)** — Recycle flow from clarifier to aeration  
• **F‑TKN (mg/L)** — Final‑effluent *Total Kjeldahl Nitrogen*  
• **F‑NO₃ (mg/L)** — Final‑effluent *Nitrate‑N*  
• **SRT (days)** — *Sludge Retention Time*  
• **Flow (L/s)** — Influent hydraulic flow to the plant  
• **Flow‑Aeration (L/s)** — Aeration air/oxygen flow (blower proxy)  
• **DO (mg/L)** — Dissolved Oxygen in aeration basin  
• **F‑pH (–)** — Final‑effluent pH  
• **Temperature (°C)** — Mixed liquor/effluent temperature

Output
------
• **F‑BOD (mg/L)** — Predicted *Biological Oxygen Demand* in final effluent

Reproducible setup
------------------
**Conda (preferred)**:  
    conda env create -f Python_Environment/AI_BOD_Softsenor_PyEnv.yaml
    conda activate AI_BOD_Softsenor_PyEnv

**Pip (alternative)**:  
    pip install -r requirements

Model file sizes & Git LFS
--------------------------
If `NNmodel.keras` exceeds 100 MB, track with **Git LFS**:
    git lfs install
    git lfs track "*.keras"
    git add .gitattributes NN_AI-model/NNmodel.keras
    git commit -m "Track model with Git LFS"
    git push

Cite
----

By using this dataset, tool/script or the model in research/industry, you acknowledge and agree that proper citation is required. If you incorporate or reference this dataset in your work, please cite the following:

Hassnain, M.; Lee, S.M.W.; Azhar, M.R. (2025).
Soft Sensing of Biological Oxygen Demand in Industrial Wastewater Using Machine Learning Models.
SSRN: https://ssrn.com/abstract=5360563 | DOI: http://dx.doi.org/10.2139/ssrn.5360563

Failure to provide proper citation may violate academic and intellectual property rights.
