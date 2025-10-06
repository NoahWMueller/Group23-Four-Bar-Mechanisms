# Four-Bar Linkage Analysis Tool

This project is a **Python-based command-line tool** for the complete kinematic analysis and visualization of four-bar linkage mechanisms. It allows users to define custom linkages or use a built-in preset to:

- Classify the mechanism
- Determine its range of motion
- Plot the complex curve traced by a point on its coupler link

---

## Features

- **Mechanism Classification**  
  Automatically applies **Grashof's criterion** to classify the linkage (e.g., Crank-Rocker, Drag-Link, etc.).

- **Kinematic Solver**  
  Accurately calculates the linkage's geometry for both *open* and *crossed* assembly configurations using a robust solver.

- **Coupler Curve Plotting**  
  Generates and displays a precise plot of the coupler curve for any point of interest on the coupler link, including custom offsets.

- **Motion Range Analysis**  
  Automatically determines the feasible input angle ranges for mechanisms that cannot complete a full rotation.

- **Interactive Interface**  
  A user-friendly command-line interface guides you through the process of defining your mechanism.

- **Preset Demo**  
  Includes a built-in demo to quickly showcase the tool's capabilities.

---

## Requirements & Installation

This script requires **Python 3** and a couple of common scientific computing libraries.

- **Python 3**: Make sure you have Python installed.
- **Required Libraries**: `numpy` and `matplotlib`

Install the required libraries with:

```bash
pip install numpy matplotlib
```

## How to Use

1.  **Download** the script file (e.g., `coding_project.py`) to a directory on your computer.

2.  **Navigate** to that directory in your terminal.

3.  **Run** the script with the following command:
    ```bash
    python coding_project.py
    ```
4.  **Follow the prompts** that appear in the terminal:
    -   Choose **Option 1** to design your own system.
    -   Choose **Option 2** to use the preset demo.

    If you select **Option 1**, you will be asked to enter the four link lengths, the parallel and perpendicular offsets for the coupler point, the assembly mode (`Open`, `Crossed`, or `Both`), and the ground link's orientation angle.

5.  The program will then output the mechanism's classification and display a plot of the coupler curve.

---

## 🎥 Demonstration: Preset Mechanism

To see the program in action without entering custom values, choose **Option 2** when prompted.

This will load a pre-defined **Crank-Rocker** mechanism and generate its coupler curve for a specific offset point, providing a great way to quickly verify that the script is working correctly.
