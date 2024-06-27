import numpy as np
import matplotlib.pyplot as plt

def hill_function(x1, x2, h1, h2):
    return 1 / (1 + x1*h1 + x2*h2)

def differential_equation(x1, x2, t, K, r, τ1, τ2, h1, h2, delayed_values1, delayed_values2, dt):
    delayed_index1 = int((t - τ1) / dt)
    delayed_index2 = int((t - τ2) / dt)
    delayed_term1 = 0.0 if delayed_index1 < 0 or delayed_index1 >= len(delayed_values1) else delayed_values1[delayed_index1]
    delayed_term2 = 0.0 if delayed_index2 < 0 or delayed_index2 >= len(delayed_values2) else delayed_values2[delayed_index2]
    dx1_dt = K * hill_function(delayed_term1, delayed_term2, h1, h2) - r * x1
    dx2_dt = K * hill_function(delayed_term2, delayed_term1, h1, h2) - r * x2
    return dx1_dt, dx2_dt

x10 = 0.1
x20 = 0.2
K = 5.0
r = 0.1
h1 = 2
h2 = 3
τ1 = 50
τ2 = 16.5

dt = 0.01
t_horizon = 1000

t = np.arange(0, t_horizon, dt)
x1 = np.zeros(t.shape)
x2 = np.zeros(t.shape)
delayed_values1 = np.zeros(t.shape)
delayed_values2 = np.zeros(t.shape)
x1[0] = x10
x2[0] = x20

for i in range(1, len(t)):
    delayed_values1[i] = x1[i-1]
    delayed_values2[i] = x2[i-1]
    dx1_dt, dx2_dt = differential_equation(x1[i-1], x2[i-1], t[i-1], K, r, τ1, τ2, h1, h2, delayed_values1, delayed_values2, dt)
    x1[i] = x1[i-1] + dt * dx1_dt
    x2[i] = x2[i-1] + dt * dx2_dt

# Decrease the figure size
plt.figure(figsize=(6, 4))

plt.plot(t, x1, label='x1', linestyle='-', color='blue')
plt.plot(t, x2, label='x2', linestyle='-.', color='red')
plt.xlabel('Time')
plt.ylabel('x')
plt.title('Solution of the differential equations')
plt.show()