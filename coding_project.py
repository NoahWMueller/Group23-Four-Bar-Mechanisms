import numpy as np
import matplotlib.pyplot as plt

# ==========================================================================================
# 4-BAR LINKAGE ANALYSIS PROGRAM
# ==========================================================================================
# This program:
# 1. Takes link lengths, coupler offsets, and assembly mode from the user
# 2. Classifies the linkage using Grashof’s criterion
# 3. Finds feasible input angle ranges (theta2_min and theta2_max)
# 4. Plots the coupler curve of the mechanism
# 5. Issues warnings if the linkage is impossible
# ==========================================================================================

# ------------------------------------------------------------------------------------------
def solve_linkage(r1, r2, r3, r4, theta2):
    """
    Solves the four-bar linkage using the Law of Cosines to find the
    angles of the coupler (theta3) and output (theta4) links.

    Parameters:
        r1, r2, r3, r4 (float): Link lengths (Ground, Input, Coupler, Output)
        theta2 (float): Input angle (radians)

    Returns:
        tuple: A tuple containing two solution tuples for (theta3, theta4):
               ((theta3_open, theta4_open), (theta3_crossed, theta4_crossed))
               Returns (None, None) if no real solution exists.
    """
    try:
        # Position of input pivot B
        Bx = r2 * np.cos(theta2)
        By = r2 * np.sin(theta2)

        # Vector from B to ground pivot D
        Dx = r1
        BD_vec_x = Dx - Bx
        BD_vec_y = 0 - By
        c_squared = BD_vec_x**2 + BD_vec_y**2
        c = np.sqrt(c_squared)

        # Check for impossible configurations (triangle inequality)
        if c > r3 + r4 or c < abs(r3 - r4):
            return None, None

        # Law of Cosines on triangle BCD to find angle 'beta' at B
        # Use np.clip to handle floating-point inaccuracies
        beta = np.arccos(np.clip((r3**2 + c_squared - r4**2) / (2 * r3 * c), -1.0, 1.0))

        # Angle of vector BD with x-axis
        gamma = np.arctan2(BD_vec_y, BD_vec_x)

        # Open solution
        theta3_open = gamma + beta
        theta4_open = np.arctan2(r2 * np.sin(theta2) + r3 * np.sin(theta3_open),
                                 r2 * np.cos(theta2) + r3 * np.cos(theta3_open) - r1)

        # Crossed solution
        theta3_crossed = gamma - beta
        theta4_crossed = np.arctan2(r2 * np.sin(theta2) + r3 * np.sin(theta3_crossed),
                                   r2 * np.cos(theta2) + r3 * np.cos(theta3_crossed) - r1)

        return ((theta3_open, theta4_open), (theta3_crossed, theta4_crossed))
    except (ValueError, ZeroDivisionError):
        # Handle cases where arccos input is out of range or division by zero occurs
        return None, None

# ------------------------------------------------------------------------------------------
def find_theta_limits(r1, r2, r3, r4, ngrid=2000):
    """
    Finds feasible input link angles (theta2) for the 4-bar linkage.

    Parameters:
        r1, r2, r3, r4 (float): Link lengths
        ngrid (int): Number of theta2 samples to test (resolution)

    Returns:
        list: List of intervals [(theta2_start, theta2_end), ...] in radians
              Empty list if mechanism is impossible
    """
    thetas = np.linspace(0, 2 * np.pi, ngrid, endpoint=False)
    feasible = np.zeros(ngrid, dtype=bool)

    # Simple check for general impossibility based on triangle inequality
    links = np.array([r1, r2, r3, r4])
    if np.max(links) >= np.sum(links) - np.max(links):
        print("    WARNING: Impossible linkage based on the triangle inequality.")
        return []

    for i, th2 in enumerate(thetas):
        sols = solve_linkage(r1, r2, r3, r4, th2)
        if sols[0] is not None:
            feasible[i] = True

    idx = np.where(feasible)[0]
    if len(idx) == 0:
        print("    WARNING: No feasible input angles found: mechanism cannot move.")
        return []

    intervals = []
    
    # Handle wrap-around case: if feasible region spans 0/360 degrees
    if idx[0] == 0 and idx[-1] == ngrid - 1:
        # Find the gap
        diffs = np.diff(idx)
        gap_idx = np.where(diffs > 1)[0]
        if len(gap_idx) > 0:
            split_point = gap_idx[0] + 1
            idx = np.concatenate((idx[split_point:], idx[:split_point]))
    
    # Identify continuous intervals
    if len(idx) > 0:
        start_idx = idx[0]
        for i in range(1, len(idx)):
            if idx[i] != idx[i-1] + 1:
                intervals.append((thetas[start_idx], thetas[idx[i-1]]))
                start_idx = idx[i]
        intervals.append((thetas[start_idx], thetas[idx[-1]]))

    return intervals

# ------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------
def coupler_curve_geom(r1, r2, r3, r4, offset, assembly="open", nsteps=500):
    """
    Computes and plots the coupler curve of a 4-bar linkage.

    Parameters:
        r1, r2, r3, r4 (float): Link lengths
        offset (dict): {"parallel": value, "perpendicular": value} - offsets from coupler link
        assembly (str): "open", "crossed", or "both"
        nsteps (int): Number of points for the plot

    Returns:
        list: List of coupler point coordinates [(x, y), ...]
    """
    points = []
    theta_intervals = find_theta_limits(r1, r2, r3, r4, ngrid=5000)
    if not theta_intervals:
        print("    WARNING: Coupler curve cannot be computed.")
        return points

    plt.figure(figsize=(8, 8))

    for a, b in theta_intervals:
        print(f"Feasible θ₂ range: {np.degrees(a):.2f}° to {np.degrees(b):.2f}°")
        thetas2 = np.linspace(a, b, nsteps)

        # Helper function to compute coupler point from the midpoint
        def get_coupler_point(theta2_val, theta3_val):
            B = np.array([r2 * np.cos(theta2_val), r2 * np.sin(theta2_val)])
            C = B + np.array([r3 * np.cos(theta3_val), r3 * np.sin(theta3_val)])
            
            # Calculate the midpoint of the coupler link (segment BC)
            midpoint = (B + C) / 2
            
            BC_vec = C - B
            if np.linalg.norm(BC_vec) == 0:
                return B  # Handle degenerate case
            
            u = BC_vec / np.linalg.norm(BC_vec)
            n = np.array([-u[1], u[0]])
            
            # Apply offsets from the midpoint
            P = midpoint + offset['parallel'] * u + offset['perpendicular'] * n
            return P

        # Initialize the previous point for continuity check
        prev_point = None

        for i, theta2 in enumerate(thetas2):
            solutions = solve_linkage(r1, r2, r3, r4, theta2)
            if solutions is None or not solutions[0] or not solutions[1]:
                continue
            
            open_sol, crossed_sol = solutions

            if assembly.lower() == "both":
                # For 'both', we simply add both solutions
                if open_sol[0] is not None:
                    points.append(get_coupler_point(theta2, open_sol[0]))
                if crossed_sol[0] is not None:
                    points.append(get_coupler_point(theta2, crossed_sol[0]))
            
            else:
                # For 'open' or 'crossed', choose the solution closest to the previous point
                current_open_point = get_coupler_point(theta2, open_sol[0])
                current_crossed_point = get_coupler_point(theta2, crossed_sol[0])

                if i == 0:
                    # For the first point, use the assembly mode to choose
                    if assembly.lower() == "open":
                        prev_point = current_open_point
                        points.append(prev_point)
                    else: # assembly is "crossed"
                        prev_point = current_crossed_point
                        points.append(prev_point)
                else:
                    # For subsequent points, pick the one closest to the last point
                    dist_open = np.linalg.norm(current_open_point - prev_point)
                    dist_crossed = np.linalg.norm(current_crossed_point - prev_point)

                    if dist_open < dist_crossed:
                        prev_point = current_open_point
                        points.append(prev_point)
                    else:
                        prev_point = current_crossed_point
                        points.append(prev_point)

    if points:
        pts = np.array(points)
        plt.scatter(pts[:, 0], pts[:, 1], s=5, color='blue')
        plt.xlabel("x")
        plt.ylabel("y")
        plt.axis("equal")
        plt.title(f"Coupler Curve ({assembly})")
        plt.show()

    return points

# ------------------------------------------------------------------------------------------
def take_inputs():
    """
    Prompts user for input values: link lengths, offsets, and assembly mode.

    Returns:
        dict: Link lengths {"Ground Link", "Input Link", "Coupler Link", "Output Link"}
        dict: Offsets {"parallel", "perpendicular"}
        str: Assembly mode ("Open", "Crossed", "Both")
    """
    Coupler_lengths = {"Ground Link": 0, "Input Link": 0, "Coupler Link": 0, "Output Link": 0}
    for link in Coupler_lengths.keys():
        while True:
            try:
                val = float(input(f"Enter {link} length: "))
                if val > 0:
                    Coupler_lengths[link] = val
                    break
                else:
                    print("Link length must be a positive number.")
            except ValueError:
                print("Invalid input, please enter a number.")

    Coupler_offsets = {"parallel": 0, "perpendicular": 0}
    for key in Coupler_offsets.keys():
        while True:
            try:
                val = float(input(f"Enter coupler offset {key} to coupler link: "))
                Coupler_offsets[key] = val
                break
            except ValueError:
                print("Invalid input, please enter a number.")

    while True:
        Assembly_mode = input("Choose assembly mode [Open/Crossed/Both]: ").capitalize()
        if Assembly_mode in ["Open", "Crossed", "Both"]:
            break
        else:
            print("Invalid option, try again.")

    return Coupler_lengths, Coupler_offsets, Assembly_mode


# ------------------------------------------------------------------------------------------
def classify_mechanism(Coupler_lengths):
    """
    Classifies the linkage according to Grashof’s criterion.

    Parameters:
        Coupler_lengths (dict): {"Ground Link", "Input Link", "Coupler Link", "Output Link"}

    Returns:
        str: Classification ("Double-crank", "Crank-rocker", "Double-rocker", "Non-Grashof")
    """
    L = sorted(list(Coupler_lengths.values()))
    s, p, q, l = L[0], L[1], L[2], L[3]
    if s + l <= p + q:
        print("Parameters satisfy Grashof's Criterion.")
        grashof = True
    else:
        print("Parameters do not satisfy Grashof's Criterion.")
        grashof = False

    if not grashof:
        return "Non-Grashof (Triple-rocker)"

    # Identify configuration by which link is the shortest
    if Coupler_lengths['Ground Link'] == s:
        return "Double-crank (Drag-link)"
    elif Coupler_lengths['Input Link'] == s or Coupler_lengths['Output Link'] == s:
        return "Crank-rocker"
    elif Coupler_lengths['Coupler Link'] == s:
        return "Double-rocker (Rocker-rocker)"
    else:
        return "Grashof, but special case."

# ------------------------------------------------------------------------------------------
def main():
    """
    Main program flow:
    1. Get inputs from user
    2. Classify mechanism
    3. Compute and plot coupler curve
    """
    while True:
        demo = input(f"Option 1: Design your own system.\nOption 2: Use preset for Vertical demo.\nAnswer: ")
        if demo in ["1", "2"]:
            break
        else:
            print("Please answer with 1 or 2.")

    if demo == "2":
        # This preset is a classic Grashof crank-rocker linkage
        Coupler_lengths = {"Ground Link": 1, "Input Link": 1, "Coupler Link": 1, "Output Link": 1}
        Coupler_offsets = {"parallel": 0, "perpendicular": 0}
        Assembly_mode = "Both"
        mech_type = classify_mechanism(Coupler_lengths)
        print(f"Mechanism classification: {mech_type}")
    else:
        Coupler_lengths, Coupler_offsets, Assembly_mode = take_inputs()
        mech_type = classify_mechanism(Coupler_lengths)
        print(f"Mechanism classification: {mech_type}")

    coupler_curve_geom(
        Coupler_lengths['Ground Link'],
        Coupler_lengths["Input Link"],
        Coupler_lengths["Coupler Link"],
        Coupler_lengths["Output Link"],
        Coupler_offsets,
        Assembly_mode,
        nsteps=1000
    )


# ------------------------------------------------------------------------------------------
if __name__ == "__main__":
    main()