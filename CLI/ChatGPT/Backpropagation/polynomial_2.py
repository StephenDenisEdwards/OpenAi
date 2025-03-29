import numpy as np
import matplotlib.pyplot as plt

# Sample data
x = np.array([1, 2.4, 3, 4, 5.5, 6.2, 8])
y = np.array([2.1, 3.9, 8.2, 14.8, 30.5, 55.1, 91.2])

# Fit a degree 5 polynomial
coeffs = np.polyfit(x, y, 5)

# Create polynomial function from the coefficients
p = np.poly1d(coeffs)

# Plot
x_fit = np.linspace(min(x), max(x), 200)
y_fit = p(x_fit)

plt.scatter(x, y, color='red', label='Data')
plt.plot(x_fit, y_fit, label='Degree 5 Fit')
plt.legend()
plt.show(block=True)
