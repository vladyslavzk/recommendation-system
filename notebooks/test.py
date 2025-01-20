import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime


# Load one month of data to start
df = pd.read_csv('2019-Oct.csv')

# Basic information about our dataset
print("Dataset Overview:")
print(f"Number of records: {len(df):,}")
print(f"Memory usage: {df.memory_usage().sum() / 1024**2:.2f} MB")
print("\nColumns and their types:")
df.info()