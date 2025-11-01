# In control/ga_optimizer.py

from deap import base, creator, tools, algorithms
import random
import numpy as np
from tensorflow.keras.models import load_model
import os
import pickle

# --- 1. Load the Model AND the Scaler ---
# Get file paths from control/ann_model.py
from .ann_model import MODEL_FILE

SCALER_FILE = os.path.join(os.path.dirname(__file__), '..', 'data', 'scaler.pkl')

if not os.path.exists(MODEL_FILE) or not os.path.exists(SCALER_FILE):
    print(f"FATAL ERROR: Model or Scaler not found.")
    print(f"Please run `train_model.py` first.")
    exit()

print(f"GA Optimizer: Loading model from {MODEL_FILE}...")
ANN_MODEL = load_model(MODEL_FILE)
print("GA Optimizer: Model loaded.")

print(f"GA Optimizer: Loading scaler from {SCALER_FILE}...")
SCALER = pickle.load(open(SCALER_FILE, 'rb'))
print("GA Optimizer: Scaler loaded.")
# ----------------------------------------

# --- Global variables for the GA evaluation ---
# These will be set by main.py
PARAM_RANGES = {}
STATIC_INPUTS = []  # This will hold our known diameter

# 1. Define Fitness
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))

# 2. Define Chromosome (Individual)
# The individual will only contain the 3 parameters we are optimizing
creator.create("Individual", list, fitness=creator.FitnessMin)

toolbox = base.Toolbox()

# 3. Define Genes (Attributes)
# We will register these dynamically from main.py
toolbox.register("attr_pressure", random.uniform, 100, 240)
toolbox.register("attr_flow_rate", random.uniform, 0.07, 0.33)
toolbox.register("attr_traverse_rate", random.uniform, 30, 150)

# 4. Define Individual and Population
toolbox.register(
    "individual",
    tools.initCycle,
    creator.Individual,
    (toolbox.attr_pressure, toolbox.attr_flow_rate, toolbox.attr_traverse_rate),
    n=1
)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)


# 5. Define the Evaluation (Fitness) Function
def evaluate_fitness(individual):
    """
    This function takes the 3 optimizable parameters,
    combines them with the static (known) diameter,
    scales them, and predicts the depth.
    """
    pressure, flow_rate, traverse_rate = individual

    # Combine with the static diameter input
    # Order must match the training script: [P, m_a, v, d]
    raw_inputs = np.array(
        [pressure, flow_rate, traverse_rate] + STATIC_INPUTS
    ).reshape(1, -1)

    # --- THIS IS THE CRITICAL FIX ---
    # Scale the inputs using the saved scaler
    scaled_inputs = SCALER.transform(raw_inputs)
    # --------------------------------

    # Predict using the scaled inputs
    predicted_depth = ANN_MODEL.predict(scaled_inputs, verbose=0)

    error = abs(predicted_depth[0, 0] - DESIRED_DEPTH_OF_CUT)

    return (error,)


# 6. Register Genetic Operators
toolbox.register("evaluate", evaluate_fitness)
toolbox.register("mate", tools.cxSimulatedBinaryBounded, eta=20.0, low=[], up=[])
toolbox.register("mutate", tools.mutPolynomialBounded, eta=20.0, low=[], up=[], indpb=0.1)
toolbox.register("select", tools.selTournament, tournsize=3)


def run_genetic_algorithm(param_ranges, static_inputs, desired_depth):
    """
    Runs the full GA optimization.
    """
    global PARAM_RANGES, STATIC_INPUTS, DESIRED_DEPTH_OF_CUT
    PARAM_RANGES = param_ranges
    STATIC_INPUTS = static_inputs
    DESIRED_DEPTH_OF_CUT = desired_depth

    # --- Dynamically set boundaries for GA ---
    # Get boundaries for the 3 parameters we are optimizing
    p_min, p_max = param_ranges['water_pressure']
    ma_min, ma_max = param_ranges['abrasive_flow']
    v_min, v_max = param_ranges['traverse_speed']

    low_bounds = [p_min, ma_min, v_min]
    up_bounds = [p_max, ma_max, v_min]  # Typo corrected: should be v_max

    # Re-register attributes with correct boundaries
    toolbox.register("attr_pressure", random.uniform, p_min, p_max)
    toolbox.register("attr_flow_rate", random.uniform, ma_min, ma_max)
    toolbox.register("attr_traverse_rate", random.uniform, v_min, v_max)

    # Update operator boundaries
    toolbox.unregister("mate")
    toolbox.unregister("mutate")
    toolbox.register("mate", tools.cxSimulatedBinaryBounded,
                     eta=20.0, low=low_bounds, up=up_bounds)
    toolbox.register("mutate", tools.mutPolynomialBounded,
                     eta=20.0, low=low_bounds, up=up_bounds, indpb=0.1)
    # --- End of dynamic setup ---

    POP_SIZE = 50
    CXPB = 0.9
    MUTPB = 0.1
    NGEN = 100

    pop = toolbox.population(n=POP_SIZE)
    hof = tools.HallOfFame(1)

    algorithms.eaSimple(
        pop, toolbox, cxpb=CXPB, mutpb=MUTPB, ngen=NGEN,
        halloffame=hof, verbose=True
    )

    best_individual = hof[0]
    best_params = {
        "pressure": best_individual[0],
        "flow_rate": best_individual[1],
        "traverse_rate": best_individual[2]
    }
    final_error = best_individual.fitness.values[0]

    print("--- GA Optimization Complete ---")
    print(f"Target Depth: {desired_depth:.2f} mm")
    print(f"Achieved with Error: {final_error:.4f} mm")
    print("Optimal Parameters:")
    print(f"  Water Pressure (P): {best_params['pressure']:.2f} MPa")
    print(f"  Abrasive Flow (m_a): {best_params['flow_rate']:.3f} kg/min")
    print(f"  Traverse Rate (v): {best_params['traverse_rate']:.2f} mm/min")

    return best_params