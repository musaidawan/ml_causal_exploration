import numpy as np
import pandas as pd
import statsmodels.api as sm

# Create sample data
np.random.seed(42)  # for reproducibility
study_hours = np.random.normal(5, 2, 50)  # 50 students, mean=5 hours, std=2
# Test scores will be influenced by study hours plus some random noise
test_scores = 60 + (8 * study_hours) + np.random.normal(0, 10, 50)

# Create a DataFrame
df = pd.DataFrame({
    'study_hours': study_hours,
    'test_scores': test_scores
})

# Prepare the data for statsmodels (add constant for intercept)
X = sm.add_constant(df['study_hours'])
y = df['test_scores']

# Run OLS regression
model = sm.OLS(y, X).fit()

# Print the summary
print(model.summary())

# Get specific values
print("\nCoefficients:")
print(f"Intercept: {model.params[0]:.2f}")
print(f"Study Hours Coefficient: {model.params[1]:.2f}")
