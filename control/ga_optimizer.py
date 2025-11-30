print("--- RUNNING ga_optimizer.py (5-INPUT DEEP MODEL w/ BLEND Crossover) ---")

from deap import base, creator, tools, algorithms
import random
import numpy as np
from tensorflow.keras.models import load_model
import os
import pickle

# --- 1. Load the Model AND the Scaler ---
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

PARAM_RANGES = {}
STATIC_INPUTS = []
DESIRED_DEPTH_OF_CUT = 0.0

# 1. Define Fitness
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))

# 2. Define Chromosome (Individual) - Still 3 parameters
creator.create("Individual", list, fitness=creator.FitnessMin)

toolbox = base.Toolbox()

# 3. Define Genes (Attributes) - Still 3 parameters
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
    pressure, flow_rate, traverse_rate = individual

    # Order MUST match FEATURES_LIST in train_model.py:
    # [P (MPa), mf (kg/min), v (mm/min), df (mm), do (mm)]

    raw_inputs = np.array(
        [pressure, flow_rate, traverse_rate] + STATIC_INPUTS
    ).reshape(1, -1)

    # Scale the raw 5-element input vector
    scaled_inputs = SCALER.transform(raw_inputs)

    predicted_depth = ANN_MODEL.predict(scaled_inputs, verbose=0)
    error = abs(predicted_depth[0, 0] - DESIRED_DEPTH_OF_CUT)
    return (error,)


# 6. Register Genetic Operators
toolbox.register("evaluate", evaluate_fitness)
# --- THIS IS THE ALTERATION ---
# We are swapping the buggy 'cxSimulatedBinaryBounded'
# with the more stable 'cxBlend'. alpha=0.5 is a standard value.
toolbox.register("mate", tools.cxBlend, alpha=0.5)
# ------------------------------
toolbox.register("mutate", tools.mutPolynomialBounded, eta=20.0, low=[], up=[], indpb=0.1)
toolbox.register("select", tools.selTournament, tournsize=3)


def run_genetic_algorithm(param_ranges, static_inputs, desired_depth):
    global PARAM_RANGES, STATIC_INPUTS, DESIRED_DEPTH_OF_CUT
    PARAM_RANGES = param_ranges
    STATIC_INPUTS = static_inputs
    DESIRED_DEPTH_OF_CUT = desired_depth

    # --- Use the exact column names from your CSV ---
    p_min, p_max = param_ranges['P (MPa)']
    ma_min, ma_max = param_ranges['mf (kg/min)']
    v_min, v_max = param_ranges['v (mm/min)']

    low_bounds = [p_min, ma_min, v_min]
    up_bounds = [p_max, ma_max, v_max]  # Typo is fixed

    # Re-register attributes with correct boundaries
    toolbox.register("attr_pressure", random.uniform, p_min, p_max)
    toolbox.register("attr_flow_rate", random.uniform, ma_min, ma_max)
    toolbox.register("attr_traverse_rate", random.uniform, v_min, v_max)

    # Update operator boundaries
    toolbox.unregister("mate")  # Unregister the old one
    # --- THIS IS THE ALTERATION ---
    # We register the new 'cxBlend' crossover. It doesn't need bounds.
    toolbox.register("mate", tools.cxBlend, alpha=0.5)
    # ------------------------------

    # The mutation operator *does* need bounds, so we update it
    toolbox.unregister("mutate")
    toolbox.register("mutate", tools.mutPolynomialBounded,
                     eta=2.0, low=low_bounds, up=up_bounds, indpb=0.1)

    POP_SIZE = 30
    CXPB = 0.9
    MUTPB = 0.01
    NGEN = 30

    pop = toolbox.population(n=POP_SIZE)
    hof = tools.HallOfFame(1)

    algorithms.eaSimple(
        pop, toolbox, cxpb=CXPB, mutpb=MUTPB, ngen=NGEN,
        halloffame=hof, verbose=True
    )

    best_individual = hof[0]

    # --- NEW: Safety Check ---
    # Because cxBlend doesn't respect bounds, we must manually
    # clip the final answer to be within the valid range.
    best_pressure = np.clip(best_individual[0], p_min, p_max)
    best_flow = np.clip(best_individual[1], ma_min, ma_max)
    best_traverse = np.clip(best_individual[2], v_min, v_max)
    # --- End Safety Check ---

    best_params = {
        "pressure": best_pressure,
        "flow_rate": best_flow,
        "traverse_rate": best_traverse
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

