import math
import numpy as np
import matplotlib.pyplot as plt

############################################################################################
# 4-BAR LINKAGE ANALYSIS PROGRAM
############################################################################################
#
# USAGE INSTRUCTIONS:
# 1. Run this script from your terminal (`python coding_project.py`).
# 2. Choose to enter custom linkage parameters (Option 1) or use a preset demo (Option 2).
# 3. For custom input, enter positive lengths for each of the four links (Ground, Input, Coupler, Output).
# 4. Specify parallel and perpendicular offsets for the coupler point (location of interest on the coupler link).
# 5. Select the assembly mode: "Open", "Crossed", or "Both".
# 6. Enter the ground link's orientation angle (in degrees).
# 7. The program will classify the mechanism (Grashof's criterion), determine valid input angle ranges, and plot the coupler curve.
#
############################################################################################

# ------------------------------------------------------------------------------------------
def solve_linkage(r1, r2, r3, r4, theta2):
    """
    Compute the coupler and output link angles for a four-bar linkage using the Law of Cosines.

    Args:
        r1, r2, r3, r4 (float): Link lengths (Ground, Input, Coupler, Output).
        theta2 (float): Input angle in radians.

    Returns:
        tuple: ((theta3_open, theta4_open), (theta3_crossed, theta4_crossed)),
               or (None, None) if no real solution exists for the given theta2.
    """
    try:
        # Calculate position of input pivot B
        Bx = r2 * np.cos(theta2)
        By = r2 * np.sin(theta2)

        # Vector from B to ground pivot D (fixed at (r1, 0))
        Dx = r1
        BD_vec_x = Dx - Bx
        BD_vec_y = 0 - By
        c_squared = BD_vec_x**2 + BD_vec_y**2
        c = np.sqrt(c_squared)

        # Triangle inequality: check if configuration is possible
        if c > r3 + r4 or c < abs(r3 - r4):
            return None, None

        # Law of Cosines: find angle 'beta' at B
        denominator = 2 * r3 * c
        if denominator == 0:
            return None, None  # Degenerate case
        beta = np.arccos(np.clip((r3**2 + c_squared - r4**2) / denominator, -1.0, 1.0))

        # Angle of BD vector with x-axis
        gamma = np.arctan2(BD_vec_y, BD_vec_x)

        # Open (elbow up) solution
        theta3_open = gamma + beta
        theta4_open = np.arctan2(r2 * np.sin(theta2) + r3 * np.sin(theta3_open),
                                 r2 * np.cos(theta2) + r3 * np.cos(theta3_open) - r1)

        # Crossed (elbow down) solution
        theta3_crossed = gamma - beta
        theta4_crossed = np.arctan2(r2 * np.sin(theta2) + r3 * np.sin(theta3_crossed),
                                   r2 * np.cos(theta2) + r3 * np.cos(theta3_crossed) - r1)

        return ((theta3_open, theta4_open), (theta3_crossed, theta4_crossed))
    except (ValueError, ZeroDivisionError):
        # Handle math errors (e.g., arccos out of range, division by zero)
        return None, None

# ------------------------------------------------------------------------------------------
def find_theta_limits(r1, r2, r3, r4, ngrid=2000):
    """
    Find feasible input angle ranges (theta2) for the four-bar linkage.

    Args:
        r1, r2, r3, r4 (float): Link lengths.
        ngrid (int): Number of theta2 samples to test for valid ranges.

    Returns:
        list: List of tuples for valid continuous theta2 ranges in radians,
              e.g., [(start1, end1), (start2, end2), ...]. Empty if not assemblable.
    """
    thetas = np.linspace(0, 2 * np.pi, ngrid, endpoint=False)
    feasible = np.zeros(ngrid, dtype=bool)

    # Quick check for impossible linkage (triangle inequality)
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
    
    # Identify continuous intervals in feasible theta2
    if len(idx) > 0:
        start_idx = idx[0]
        for i in range(1, len(idx)):
            if idx[i] != idx[i-1] + 1:
                intervals.append((thetas[start_idx], thetas[idx[i-1]]))
                start_idx = idx[i]
        intervals.append((thetas[start_idx], thetas[idx[-1]]))

    return intervals

# ------------------------------------------------------------------------------------------
def coupler_curve_geom(r1, r2, r3, r4, offset, assembly="open", nsteps=500, 
                       set_angles=None, ground_angle=0.0):
    """
    Compute and plot the coupler curve for a four-bar linkage.

    Args:
        r1, r2, r3, r4 (float): Link lengths.
        offset (dict): {"parallel": value, "perpendicular": value} - offsets from coupler link.
        assembly (str): "open", "crossed", or "both" (solution branch).
        nsteps (int): Number of points to plot.
        set_angles (tuple): Optional (start, end) theta2 angles in radians.
        ground_angle (float): Ground link orientation angle in radians.
    """
    points = []
    theta_intervals = None

    # Use set_angles directly if provided, otherwise compute feasible intervals
    if set_angles is not None:
        thetas2 = np.linspace(set_angles[0], set_angles[1], nsteps)
    else:
        theta_intervals = find_theta_limits(r1, r2, r3, r4, ngrid=5000)
        if not theta_intervals:
            print("    WARNING: Coupler curve cannot be computed.")
            return points

    plt.figure(figsize=(8, 8))

    # Rotation matrix for ground angle
    R = np.array([[np.cos(ground_angle), -np.sin(ground_angle)],
                  [np.sin(ground_angle),  np.cos(ground_angle)]])

    def get_coupler_point(theta2_val, theta3_val):
        # Calculate coupler point position given input and coupler angles
        B = np.array([r2 * np.cos(theta2_val), r2 * np.sin(theta2_val)])
        C = B + np.array([r3 * np.cos(theta3_val), r3 * np.sin(theta3_val)])
        # Midpoint of BC
        midpoint = (B + C) / 2
        BC_vec = C - B
        if np.linalg.norm(BC_vec) == 0:
            return R @ B  # Degenerate case: return rotated B
        # Unit vector along BC
        u = BC_vec / np.linalg.norm(BC_vec)
        # Perpendicular vector to BC
        n = np.array([-u[1], u[0]])
        # Offset from midpoint
        P = midpoint + offset['parallel'] * u + offset['perpendicular'] * n
        return R @ P   # Apply ground rotation

    if set_angles is not None:
        # Direct evaluation at specified theta2 angles
        prev_point = None
        for theta2 in thetas2:
            solutions = solve_linkage(r1, r2, r3, r4, theta2)
            if solutions is None or not solutions[0] or not solutions[1]:
                continue

            open_sol, crossed_sol = solutions
            current_open_point = get_coupler_point(theta2, open_sol[0])
            current_crossed_point = get_coupler_point(theta2, crossed_sol[0])

            if assembly.lower() == "both":
                if open_sol[0] is not None:
                    points.append(current_open_point)
                if crossed_sol[0] is not None:
                    points.append(current_crossed_point)
            else:
                if prev_point is None:
                    prev_point = current_open_point if assembly.lower()=="open" else current_crossed_point
                    points.append(prev_point)
                else:
                    dist_open = np.linalg.norm(current_open_point - prev_point)
                    dist_crossed = np.linalg.norm(current_crossed_point - prev_point)
                    if dist_open < dist_crossed:
                        prev_point = current_open_point
                    else:
                        prev_point = current_crossed_point
                    points.append(prev_point)

    else:
        # Sweep feasible intervals for theta2
        for a, b in theta_intervals:
            print(f"Feasible θ₂ range: {np.degrees(a):.2f}° to {np.degrees(b):.2f}°")
            thetas2 = np.linspace(a, b, nsteps)

            prev_point = None
            for i, theta2 in enumerate(thetas2):
                solutions = solve_linkage(r1, r2, r3, r4, theta2)
                if solutions is None or not solutions[0] or not solutions[1]:
                    continue

                open_sol, crossed_sol = solutions
                current_open_point = get_coupler_point(theta2, open_sol[0])
                current_crossed_point = get_coupler_point(theta2, crossed_sol[0])

                if assembly.lower() == "both":
                    if open_sol[0] is not None:
                        points.append(current_open_point)
                    if crossed_sol[0] is not None:
                        points.append(current_crossed_point)
                else:
                    if i == 0:
                        prev_point = current_open_point if assembly.lower()=="open" else current_crossed_point
                        points.append(prev_point)
                    else:
                        dist_open = np.linalg.norm(current_open_point - prev_point)
                        dist_crossed = np.linalg.norm(current_crossed_point - prev_point)
                        if dist_open < dist_crossed:
                            prev_point = current_open_point
                        else:
                            prev_point = current_crossed_point
                        points.append(prev_point)

    if points:
        pts = np.array(points)
        plt.scatter(pts[:, 0], pts[:, 1], color='blue', s=5) # Use plot for continuous lines
        plt.scatter(pts[0, 0], pts[0, 1], color='green', s=50, zorder=5, label="Start")
        plt.scatter(pts[-1, 0], pts[-1, 1], color='red', s=50, zorder=5, label="End")
        plt.xlabel("x-coordinate")
        plt.ylabel("y-coordinate")
        plt.axis("equal")
        plt.grid(True)
        plt.title(f"Coupler Curve (Assembly: {assembly}, Ground Angle: {np.degrees(ground_angle):.1f}°)")
        plt.legend()
        plt.show()

    return

# ------------------------------------------------------------------------------------------
def take_inputs():
    """
    Prompts the user for all necessary input values for the linkage.

    Returns:
        tuple: (link_lengths, coupler_offsets, assembly_mode, ground_angle)
    """
    link_lengths = {}
    link_names = ["Ground Link", "Input Link", "Coupler Link", "Output Link"]
    for name in link_names:
        while True:
            try:
                val = float(input(f"Enter {name} length: "))
                if val > 0:
                    link_lengths[name] = val
                    break
                else:
                    print("Error: Link length must be a positive number.")
            except ValueError:
                print("Error: Invalid input. Please enter a number.")

    coupler_offsets = {}
    for direction in ["parallel", "perpendicular"]:
        while True:
            try:
                val = float(input(f"Enter coupler offset {direction} to coupler link: "))
                coupler_offsets[direction] = val
                break
            except ValueError:
                print("Error: Invalid input. Please enter a number.")

    while True:
        assembly_mode = input("Choose assembly mode [Open/Crossed/Both]: ").capitalize()
        if assembly_mode in ["Open", "Crossed", "Both"]:
            break
        else:
            print("Error: Invalid option. Please enter Open, Crossed, or Both.")

    while True:
        try:
            angle_deg = float(input("Enter ground link angle in degrees [0-360]: "))
            ground_angle = math.radians(angle_deg)
            break
        except ValueError:
            print("Error: Invalid input. Please enter a number.")
            
    return link_lengths, coupler_offsets, assembly_mode, ground_angle


# ------------------------------------------------------------------------------------------
def classify_mechanism(link_lengths):
    """
    Classifies the linkage type based on Grashof’s criterion.

    Parameters:
        link_lengths (dict): Dictionary of link lengths.

    Returns:
        str: The classification of the mechanism (e.g., "Crank-rocker").
    """
    lengths = list(link_lengths.values())
    s = min(lengths)  # Shortest link
    l = max(lengths)  # Longest link
    p, q = sorted(lengths)[1:3] # The other two links

    if s + l <= p + q:
        print("This is a Grashof linkage (at least one link can fully rotate).")
        # Identify the specific type based on which link is shortest
        if link_lengths['Ground Link'] == s:
            return "Double-crank (or Drag-link)"
        elif link_lengths['Input Link'] == s:
            return "Crank-rocker"
        elif link_lengths['Output Link'] == s:
            # If output is shortest, it's a crank-rocker, but driven from the other side
            return "Crank-rocker (driven from adjacent link)"
        elif link_lengths['Coupler Link'] == s:
            return "Double-rocker"
    else:
        print("This is a non-Grashof linkage (no link can fully rotate).")
        return "Triple-rocker"

    return "Grashof, but a special case (e.g., two links have the same shortest length)."

# ------------------------------------------------------------------------------------------
def main():
    """
    Main program execution function. Handles user interaction and runs analysis.
    """
    # Prompt for demo or custom input
    while True:
        demo = input(f"Option 1: Design your own system.\nOption 2: Use preset for Vertical demo.\nAnswer: ")
        if demo in ["1", "2"]:
            break
        else:
            print("Please answer with 1 or 2.")

    if demo == "2":
        # Preset demo parameters
        Coupler_lengths = {"Ground Link": 20, "Input Link": 15, "Coupler Link": 11, "Output Link": 15}
        Coupler_offsets = {"parallel": 0, "perpendicular": -22}
        angles = [0.85,1.55]
        ground_angle = 1.57
        Assembly_mode = "Open"
        mech_type = classify_mechanism(Coupler_lengths)
        print(f"Mechanism classification: {mech_type}")
    else:
        # Custom user input
        Coupler_lengths, Coupler_offsets, Assembly_mode, ground_angle = take_inputs()
        mech_type = classify_mechanism(Coupler_lengths)
        print(f"Mechanism classification: {mech_type}")
        angles = None

    # Run coupler curve analysis and plot
    coupler_curve_geom(
        Coupler_lengths['Ground Link'],
        Coupler_lengths["Input Link"],
        Coupler_lengths["Coupler Link"],
        Coupler_lengths["Output Link"],
        Coupler_offsets,
        Assembly_mode,
        nsteps=1000,
        set_angles=angles,
        ground_angle=ground_angle
    )


# ------------------------------------------------------------------------------------------
if __name__ == "__main__":
    main()