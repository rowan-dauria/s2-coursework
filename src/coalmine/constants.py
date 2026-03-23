"""Physical constants for the coal mining accident dataset."""

import pandas as pd

N = 191  # total number of explosions (10+ killed each)
L = 40_550  # total observation period in days (15 Mar 1851 – 22 Mar 1962)
START_DATE = pd.Timestamp("1851-03-15")
ALPHA = 1  # Gamma prior shape parameter
BETA = 200  # Gamma prior rate parameter (days)
K_MAX = 30  # maximum number of change points
LAMBDA_K = 3  # Poisson prior mean for number of change points
SEED = 42
