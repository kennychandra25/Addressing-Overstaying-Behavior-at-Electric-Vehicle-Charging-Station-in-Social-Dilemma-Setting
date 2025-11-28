import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from scipy.integrate import cumtrapz
from scipy.interpolate import interp1d

np.random.seed(42)

def NREL_workplace_arrival():
    # Hourly average arrival rates
    hourly_rates = np.array([0., 0., 0., 0., 1., 1., 3., 10., 14., 5., 2., 2., 4.,
                             3., 2., 1., 1., 1., 1., 1., 1., 1., 0., 1.])
    
    # Create minute-based data
    minute_data = []
    for hour, rate in enumerate(hourly_rates):
        arrivals = int(rate * 60)  # Approximate arrivals for the hour, scaled to minutes
        minute_data.extend(np.random.uniform(low=hour*60, high=(hour+1)*60, size=arrivals))
    
    minute_data = np.array(minute_data)
    
    # Perform Kernel Density Estimation (KDE)
    kde = gaussian_kde(minute_data)
    x_grid = np.linspace(0, 24*60, 1440)  # 24 hours in minute units
    pdf_values = kde(x_grid)
    
    # Compute the Cumulative Distribution Function (CDF)
    cdf_values = cumtrapz(pdf_values, x_grid, initial=0)
    cdf_values /= cdf_values[-1]  # Normalize to ensure the CDF ends at 1
    
    # Interpolate the inverse CDF
    inverse_cdf = interp1d(cdf_values, x_grid, bounds_error=False, fill_value=(x_grid[0], x_grid[-1]))
    
    # Simulate arrival times as a random process
    num_samples = 55  # Number of arrivals to simulate
    random_probabilities = np.random.uniform(0, 1, num_samples)
    arrival_times = inverse_cdf(random_probabilities)
    
    # Align arrival times with discrete time steps
    time_step = 1  # Simulation time step in minutes
    discrete_arrival_counts = np.zeros(int(24*60/time_step))  # 24 hours in minute steps
    
    for t in arrival_times:
        index = int(t // time_step)
        discrete_arrival_counts[index] += 1
    
    # Plot the arrival process
    # plt.figure(figsize=(10, 6))
    # plt.plot(np.arange(len(discrete_arrival_counts)) * time_step, discrete_arrival_counts, label="Discrete Arrival Counts")
    # plt.xlabel("Time (minutes)")
    # plt.ylabel("Number of Arrivals")
    # plt.title("Simulated Arrival Process")
    # plt.legend()
    # plt.grid(True)
    # plt.show()
    
    return discrete_arrival_counts

# Hourly average arrival rates
hourly_rates = np.array([0., 0., 0., 0., 1., 1., 3., 10., 14., 5., 2., 2., 4.,
                          3., 2., 1., 1., 1., 1., 1., 1., 1., 0., 1.])

# Create minute-based data
minute_data = []
for hour, rate in enumerate(hourly_rates):
    arrivals = int(rate * 60)  # Approximate arrivals for the hour, scaled to minutes
    minute_data.extend(np.random.uniform(low=hour*60, high=(hour+1)*60, size=arrivals))

minute_data = np.array(minute_data)

# Perform Kernel Density Estimation (KDE)
kde = gaussian_kde(minute_data)
x_grid = np.linspace(0, 24*60, 1440)  # 24 hours in minute units
pdf_values = kde(x_grid)

# Compute the Cumulative Distribution Function (CDF)
cdf_values = cumtrapz(pdf_values, x_grid, initial=0)
cdf_values /= cdf_values[-1]  # Normalize to ensure the CDF ends at 1

# Interpolate the inverse CDF
inverse_cdf = interp1d(cdf_values, x_grid, bounds_error=False, fill_value=(x_grid[0], x_grid[-1]))

# Simulate arrival times as a random process
num_samples = 55  # Number of arrivals to simulate
random_probabilities = np.random.uniform(0, 1, num_samples)
arrival_times = inverse_cdf(random_probabilities)

# Align arrival times with discrete time steps
time_step = 1  # Simulation time step in minutes
discrete_arrival_counts = np.zeros(int(24*60/time_step))  # 24 hours in minute steps

for t in arrival_times:
    index = int(t // time_step)
    discrete_arrival_counts[index] += 1

# Plot the arrival process
plt.figure(figsize=(10, 6))
plt.plot(np.arange(len(discrete_arrival_counts)) * time_step, discrete_arrival_counts, label="Discrete Arrival Counts")
plt.xlabel("Time-of-day (minutes)")
plt.ylabel("Number of EV Arrivals")
plt.title("Simulated Arrival Process")
plt.legend()
plt.grid(True)
plt.savefig(r"C:\Users\kenny\OneDrive - purdue.edu\Documents\Kenny's File\Transportation Literature\Fall 2024\CE 597 SET\Project\Workplace charging data\arrival_kde.png",
           dpi = 900)
plt.show()
