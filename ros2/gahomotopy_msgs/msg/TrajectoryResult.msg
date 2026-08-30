# Result trajectory from the GA node after optimization.
# Published on /trajectory topic.

# --- Joint-space trajectory ---
# Each row is a configuration [q1, q2, ..., qn] in degrees
# Flattened row-major: trajectory[i * num_var + j]
float64[] trajectory
int32 num_var                # number of joints (3 or 6)
int32 num_points             # number of waypoints

# --- Optimization result ---
bool success                 # true if a valid path was found
float64 path_distance        # total path distance (fitness metric)
float64 final_lambda         # final homotopy parameter (should be ~1.0 on success)

# --- Best EHPPM parameters found ---
float64 radius
float64[] obstacle_values
int32[] obstacle_signs
float64[] a_matrix           # row-major n x n

# --- GA metadata ---
float64 elapsed_time         # optimization time in seconds
float64[] fitness_evolution # best fitness per generation
string experiment_name