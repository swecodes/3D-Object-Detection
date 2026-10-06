 3D Object Detection with ResNet Scaling Analysis

This project implements a multi-scale 3D object detector on the Argoverse Tracking dataset, using ResNet backbones (depths 18/34/50/101) with varying width multipliers (0.5×/1.0×/2.0×). The notebook benchmarks FLOPs, inference latency, memory usage, and detection accuracy (mAP), and produces a roofline performance model for a T4 GPU.
⸻

Environment Setup

Option A — Google Colab (Recommended)

This notebook was designed to run on Google Colab with a T4 GPU. All setup steps below assume Colab unless noted otherwise.

1. Open Google Colab.
2. Upload msml_605code.ipynb via File → Upload notebook.
3. Enable GPU: go to Runtime → Change runtime type → T4 GPU → Save.
4. Mount your Google Drive when prompted by the first cell (required for dataset access and saving checkpoints).

Option B — Local Machine (Advanced)

A local setup requires a CUDA-capable GPU and the following prerequisites:

- Python 3.9 or higher
- CUDA 11.8 or higher
- pip and optionally conda

Create and activate a virtual environment:

python -m venv venv
source venv/bin/activate          # Linux / macOS
# OR
venv\Scripts\activate             # Windows

⸻
Installing Dependencies

On Google Colab

The first two cells of the notebook install the two non-standard packages automatically:

!pip install open3d --quiet
!pip install -U fvcore --quiet


All other dependencies (PyTorch, torchvision, OpenCV, etc.) are pre-installed in the Colab environment. No additional action is needed.

On a Local Machine

Install all dependencies from requirements.txt:

pip install -r requirements.txt


Then install PyTorch with the appropriate CUDA version for your system.
⸻

Dataset Preparation

This project uses the Argoverse HD Maps and Tracking v1.1 dataset.

1. Download argoverse-tracking_train1.zip from the Argoverse website (requires a free account).
2. Upload the zip file to your Google Drive (root of MyDrive), so the path is:
3. /content/drive/MyDrive/argoverse-tracking_train1.zip

3. The notebook will automatically extract the zip on first run to /content/argoverse_data/.

If your zip is in a different Drive folder, update the zip_path variable in the "Extract Dataset" cell accordingly.
⸻
Running the Notebook

Step-by-Step Execution

Run cells in order from top to bottom. The notebook is organized into the following stages:
Stage	Description
1. Install Packages	Installs open3d and fvcore
2. Mount Drive	Connects Google Drive for data and checkpoint I/O
3. Imports	Loads all libraries
4. Explore Dataset	Locates and inspects the extracted Argoverse sequences
5. Extract & Split	Extracts the zip and creates a 70/30 train/val split
6. Inspect Structure	Lists LiDAR, camera, and annotation files for one sequence
7. Utility Functions	Defines get_synced_pairs, load_annotations, load_image
8. Dataset & DataLoader	Builds ArgoverseDataset and DataLoader objects
9. Model Definition	Defines FPN, DetectionHead3D, and Detector3D
10. Smoke Test	Runs a dummy forward pass to verify model shapes
11. Training	Trains all 12 backbone configurations (can take 1–2 hours)
12. FLOP Counting	Measures GFLOPs and parameter counts via fvcore
13. Latency Benchmarking	Times inference for 200 runs per config
14. Profiling	Runs torch.profiler on ResNet-18/50/101 at width 1.0×
15. GPU Memory (nvidia-smi)	Queries live GPU memory after each forward pass
16. Roofline Analysis	Computes arithmetic intensity and plots the roofline model
17. Scaling Plots	Plots FLOPs vs. runtime, depth vs. runtime, memory vs. width
18. mAP Evaluation	Evaluates mAP on the validation set for all 12 configs
19. Save Results	Exports all CSVs and plots to msml605_results/ in Drive

Running All Cells at Once

In Colab, you can run the entire notebook automatically via Runtime → Run all. Be aware:

- The training stage (Stage 11) trains 12 model configurations and is the longest step. Only ResNet-101 width=0.5× is active by default (the loop is pre-configured to for depth in [101] and for wm in [0.5]). Expand the loops to train all configs if needed.
- A checkpoint is saved to Google Drive after each configuration so progress is not lost on session timeout.

Expected Outputs

After a full run, the following files are saved to /content/drive/MyDrive/msml605_results/:

flop_counts.csv
timing_results.csv
full_results.csv
roofline_analysis.csv
map_results.csv
nvidia_smi_memory.csv
profiler_results.csv
flops_vs_runtime.png
roofline_plot.png
scaling_plots.png


Checkpoints (.pth files) are saved to /content/drive/MyDrive/msml605_checkpoints/.
⸻
Troubleshooting

Drive not mounting / path not found 
Make sure you accepted the Drive permissions popup. If the path /content/drive/MyDrive/argoverse-tracking_train1.zip is not found, run the file-search cell (Stage 4) to locate the correct path and update zip_path.

CUDA out of memory 
Reduce the batch_size in the DataLoader construction cell (default is 4). Also ensure you are using a T4 GPU runtime, not CPU.

timm model not found 
Run !pip install -U timm in a new cell. The notebook uses timm to load pretrained ResNet backbones.

open3d import error 
Re-run the first install cell: !pip install open3d --quiet. On some Colab sessions, a runtime restart is required after installation — use Runtime → Restart session, then re-run all cells.

Checkpoint not loaded (random weights warning) 
This means the training stage was skipped or did not complete for that configuration. Either train that config first, or interpret mAP results as baseline random-weight performance.
