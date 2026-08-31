Gorilla Tag Movement-Cycle Analysis: Sample Data and Setup
This package contains the notebook, supporting code, and a small sample of the data required to develop and test an alternative method for dividing continuous movement into movement cycles.
1. Package contents
Please preserve the following directory structure:

gorilla_tag_sample/
├── ipynb/
│   └── Labeled_data_analyses.ipynb
├── .py/
│   ├── config_paths.py
│   └── names_format.py
├── data/
│   └── free_practice_labled_frames.csv
├── repetitive_data/
│   ├── window_predictions_walk.csv
│   ├── window_predictions_jump.csv
│   ├── window_predictions_climb.csv
│   └── processed_data/
│       ├── 002-1-s-w_pos_aligned.csv
│       ├── 002-1-s-j_pos_aligned.csv
│       ├── 002-1-s-c_pos_aligned.csv
│       ├── 002-3-e-w_pos_aligned.csv
│       ├── 002-3-e-j_pos_aligned.csv
│       └── 002-3-e-c_pos_aligned.csv
├── pixi.toml
├── pixi.lock
└── README.md
The directory structure is important because the notebook identifies the project root by locating .py/config_paths.py and then constructs all other paths relative to that location.
The two Python modules in .py are required:

• config_paths.py defines the project directory structure..
• names_format.py generates and validates the expected data filenames..
The notebook will not run correctly if only the .ipynb file is provided.
2. Sample dataset
Participant 002 is a suitable small test dataset. The sample includes:

• Three movement types: walking, jumping, and climbing..
• Day 1, start of practice (1-s)..
• Day 3, end of practice (3-e)..
• Six aligned position files in total..
• Approximately 91 MB of position data..
The filename codes are:

• w: walking.
• j: jumping.
• c: climbing.
• s: start of the practice session.
• e: end of the practice session.
For example:

002-1-s-w_pos_aligned.csv
means participant 002, Day 1, start of practice, walking task, aligned position data.
One participant is sufficient for testing file loading, visualizing the signals, and developing the cycle-segmentation procedure. For evaluating whether the method behaves consistently across participants, a sample of two or three participants would be preferable.
3. Window-prediction files
The following files define the time windows classified as walking, jumping, or climbing:

repetitive_data/window_predictions_walk.csv
repetitive_data/window_predictions_jump.csv
repetitive_data/window_predictions_climb.csv
Each row includes information such as:

• Participant number.
• Day.
• Session part.
• Task.
• Source filename.
• Window identifier.
• Start frame.
• End frame.
• Prediction.
• Prediction probability.
For the sample package, these files should be filtered so that they contain only:

• Participant 002.
• Day 1, session part s.
• Day 3, session part e.
The original column names and filenames should be preserved.
This filtering is important. If the complete prediction tables are supplied together with only six position files, the notebook will attempt to locate data for all other participants and will produce many missing-file entries.
The source_file column establishes the connection between a prediction row and its corresponding position file. For example:

source_file = 002-1-s-w
is resolved by the notebook as:

repetitive_data/processed_data/002-1-s-w_pos_aligned.csv
4. Position-data format
The aligned position files contain one row per recorded frame and columns representing the three-dimensional positions of body landmarks.
Examples include:

time
root.x
root.y
root.z
l_hand.x
l_hand.y
l_hand.z
r_hand.x
r_hand.y
r_hand.z
The files also contain positions for the head, neck, shoulders, arms, torso segments, legs, feet, and toes.
For the current movement-cycle analysis:

• Cycle detection is primarily based on the vertical hand position relative to the root:.
• r_hand.y - root.y.
• l_hand.y - root.y.
• The trajectory plots also use the x, y, and z coordinates of the hands and root..
• The number of rows in each CSV file is treated as the number of recorded frames..
The sampling frequency assumed by the notebook is:

60 Hz
Therefore:

time in seconds = frame number / 60
The package documentation should also state the physical units of the coordinates—such as millimetres or centimetres—and briefly explain how the alignment was performed. These details are not fully documented in the notebook itself but are important for interpreting amplitude, velocity, and cycle-duration measures.
5. Software environment
The project uses Pixi to manage the software environment. The supplied pixi.toml contains the required Python and Jupyter dependencies.
For exact reproducibility, it is preferable to provide both:

pixi.toml
pixi.lock
The lock file records the exact package versions used in the current environment. Without it, Pixi may resolve newer package versions when the environment is created.
From the project directory, the environment can be installed with:

pixi install
Jupyter can then be launched inside the Pixi environment:

pixi run jupyter lab
Alternatively:

pixi run jupyter notebook
The current pixi.toml is configured for:

platforms = ["win-64"]
Therefore, the supplied lock file is directly usable on 64-bit Windows. If the analysis will be run on macOS or Linux, the appropriate platform must first be added to pixi.toml, for example:

platforms = ["win-64", "linux-64"]
or:

platforms = ["win-64", "osx-arm64"]
Pixi must then regenerate the environment for that platform.
The notebook imports scikit-learn directly. Although it is present in the current lock file through the existing dependency graph, it would be safer to list it explicitly under [dependencies] in pixi.toml:

scikit-learn = "*"
6. Running the notebook
Jupyter should be launched from the main project directory:

gorilla_tag_sample/
The notebook is located at:

ipynb/Labeled_data_analyses.ipynb
The recommended procedure for the movement-cycle sample is:

1. Launch Jupyter from the project root using Pixi..
2. Open ipynb/Labeled_data_analyses.ipynb..
3. Run Section 1, “Shared setup.”.
4. Run the shared data-loading cells required to create the repetitive-practice dataframe..
5. Run the movement-cycle analysis section..
6. The other analyses do not need to be run first..
The notebook was designed so that each main analysis section is independent after the shared setup has been executed.
7. Recomputing the cycle analysis
The notebook uses cached analytical results when compatible cache files are available.
The sample package should preferably not include the existing:

img/labeled_data_analysis/cache/
directory. This ensures that the results are computed from the supplied sample files rather than loaded from a previous run.
For the first run, the following setting can also be changed in the notebook:

FORCE_RECOMPUTE["cycles"] = True
After the new cache has been created successfully, it can be returned to:

FORCE_RECOMPUTE["cycles"] = False
8. Current cycle-detection approach
The existing method operates as follows:

1. It selects windows for which prediction == 1..
2. Overlapping positive windows from the same source file are combined..
3. Short gaps of up to 15 frames are bridged..
4. Movement runs shorter than 3 seconds are excluded..
5. Hand height relative to the root is calculated..
6. The signal is smoothed using a Savitzky–Golay filter..
7. Peaks or troughs are detected, depending on the movement and hand..
8. Consecutive detected events define candidate movement cycles..
9. Candidate cycles are filtered by duration and, for walking, minimum amplitude..
10. Accepted cycles are resampled to 100 normalized time points..
The main parameters are defined in the notebook under CYCLE_PARAMS. This is the section that can be modified or replaced when testing an alternative movement-cycle segmentation method.
9. Scope of the sample
This sample is intended for:

• Reading and understanding the data format..
• Developing an alternative cycle-segmentation method..
• Testing the code from input loading through cycle extraction..
• Comparing alternative cycle boundaries with the current method..
• Producing example diagnostic plots..
It is not intended to reproduce all group-level statistical analyses in the notebook.
The complete movement-cycle analysis across the configured participants requires approximately 75 aligned position files and about 1.07 GB of data. All repetitive-practice analyses in the notebook together refer to approximately 3.21 GB of aligned position data.