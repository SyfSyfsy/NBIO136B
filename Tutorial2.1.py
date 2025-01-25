import numpy as np
import matplotlib.pyplot as plt
from numba import jit

# Set up default plotting parameters
plt.rcParams.update({
    'lines.linewidth': 2,
    'lines.markersize': 8,
    'axes.linewidth': 2,
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'axes.labelweight': 'bold',
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
})

E_r = -70e-3  # Resting potential(leak potential)
R_m = 5e6 # Membrane resistance
C_m = 2e-9  # Membrane capacitance
V_th = -50e-3  # Threshold potential
V_reset = -65e-3 # Reset potential
I_app = 4.00001e-9  #  Applied current
I_app2 = 3.999e-9  #  Applied current
#question a,b
#threshold current is 4e-9V calculated as I_app = 1/R_m *(V_th - E_r)

# print(1/R_m *(V_th - E_r) )
# Time vector
dt = 0.1e-3
t = np.arange(0, 2 + 0.1e-3, dt) # time vector
# print(t[-1])

# membrane potential vector
V_m = np.zeros_like(t)
V_m[0] = E_r

# Applied current vector
I = np.zeros_like(t)
I[0:] = I_app

# JIT-optimized simulation function
@jit(nopython=True) 

#function to simulate the neuron(LIF model)
def simulate_neuron(V_m, I, t, E_r, R_m, C_m, V_th, V_reset):

    spikes = 0
    for i in range(1, len(t)):

        V_m[i] = V_m[i - 1] + dt * (I[i-1] + 1/R_m * (E_r - V_m[i - 1])) / C_m

        if V_m[i] > V_th:

            V_m[i] = V_reset
            spikes += 1
    
    return V_m,spikes

V_m = simulate_neuron(V_m, I, t, E_r, R_m, C_m, V_th, V_reset)[0]

#Setting up another neuron with different applied current
I[0:] = I_app2
Vm_3 = np.zeros_like(t)
Vm_3[0] = E_r
Vm_3 = simulate_neuron(Vm_3, I, t, E_r, R_m, C_m, V_th, V_reset)[0]

# Plotting question 1a,1b
fig, axs = plt.subplots(2, 1, figsize=(8,6))

# Plot I = I_app2 vs Time
axs[0].plot(t,Vm_3, label='I_app = ' + str(I_app2) + 'A')
axs[0].set_title('Membrane Potential vs Time')
axs[0].set_xlabel('Time (ms)')
axs[0].set_ylabel('V_m (V)')
axs[0].grid(True)
axs[0].legend()

# Plot I = I_app vs Time
axs[1].plot(t, V_m, label='I_app = ' + str(I_app) + 'A')
axs[1].set_xlabel('Time (ms)')
axs[1].set_ylabel('V_m (V)')
axs[1].grid(True)
axs[1].legend()
plt.tight_layout()
plt.subplots_adjust(hspace=0.3)
#plt.show()

#question 1c

tau = R_m * C_m  # Time constant
I_2 = np.linspace(4e-9,5.6e-9,10)  # Applied current vector
FRR = np.zeros_like(I_2) # Firing rate vector

for i in range (0, 10):

    I = np.zeros_like(t)
    I[0:] = I_2[i]
    spikes = simulate_neuron(V_m, I, t, E_r, R_m, C_m, V_th, V_reset)[1]
    FRR[i] = spikes / 2  # Calculate firing rate in Hz

#plotting question 1c 
plt.figure(figsize=(10, 6)) 
plt.plot(I_2, FRR, 'o-')
plt.xlabel('Applied Current (1e-9A)')
plt.ylabel('Firing Rate (Hz)')
plt.title('Firing-rate Curve')
plt.grid(True)
#plt.show()


#question 1d
fr = np.zeros_like(I_2)
for i in range(0, len(I_2)):
    term1 = np.maximum(I_2[i] * R_m + E_r - V_reset, 1e-12)
    term2 = np.maximum(I_2[i] * R_m + E_r - V_th, 1e-12)
    fr[i] = 1 / (tau * (np.log(term1) - np.log(term2)))


# Plotting question 1d
plt.figure(figsize=(10, 6))

# Plot I_2 vs FRR
plt.plot(I_2, FRR, 'o-', label='By Counting Spikes', color='blue')

# Plot I_2 vs fr
plt.plot(I_2, fr, 's-', label='By equation', color='red')

plt.xlabel('Applied Current (1e-9A)')
plt.ylabel('Firing Rate (Hz)')
plt.title('Firing-rate Curve Comparison')
plt.legend()
plt.grid(True)
#plt.show()


#question 2 (a)

# JIT-optimized simulation function
@jit(nopython=True) 
#function to simulate the neuron(LIF model with noise)
def LIF_with_noise(V_m, I, t, E_r, R_m, C_m, V_th, V_reset,std_V):

    spikes = 0
    for i in range(1, len(t)):

        V_m[i] = V_m[i - 1] + dt * (I[i-1] + 1/R_m * (E_r - V_m[i - 1])) / C_m + np.random.randn()* std_V *np.sqrt(dt)

        if V_m[i] > V_th:

            V_m[i] = V_reset
            spikes += 1
    
    return V_m,spikes

#question 2 (b)

std_V1 = 0.05 #standard deviation of noise
std_V2 = 0.1
FRR1 = np.zeros_like(I_2)
FRR2 = np.zeros_like(I_2)

for i in range (0, 10):

    I = np.zeros_like(t)
    I[0:] = I_2[i]
    spikes = LIF_with_noise(V_m, I, t, E_r, R_m, C_m, V_th, V_reset,std_V1)[1]
    spikes2 = LIF_with_noise(V_m, I, t, E_r, R_m, C_m, V_th, V_reset,std_V2)[1]
    FRR1[i] = spikes / 2  # Calculate firing rate in Hz
    FRR2[i] = spikes2 / 2  # Calculate

#plotting question 2 (b)
plt.figure(figsize=(10, 6))
plt.plot(I_2, FRR, 'o-', label='No Noise')
plt.plot(I_2, FRR1, 'o-', label='STD of Noise = ' + str(std_V1))
plt.plot(I_2, FRR2, 's-', label='STD of Noise = ' + str(std_V2)) 
plt.xlabel('Applied Current (1e-9A)')
plt.ylabel('Firing Rate (Hz)')
plt.title('Firing-rate Curve')
plt.legend()
plt.grid(True)
#plt.show()


#additional plot for question 2 (b)
std_vec = np.linspace(0, 1 , 100)
FRR_vec = np.zeros_like(std_vec)
I = np.zeros_like(t)
I[0:] = 4.000001e-9
for i in range (0, 100):
    spikes = LIF_with_noise(V_m, I, t, E_r, R_m, C_m, V_th, V_reset, std_vec[i]) [1]
    FRR_vec[i] = spikes / 2  # Calculate firing rate in Hz

spike_nonoise = 1 / (tau*(np.log(4.000001e-9*R_m + E_r - V_reset) - np.log(4.000001e-9*R_m + E_r - V_th)))

#plotting additional plot for question 2 (b)
plt.figure(figsize=(10, 6))
plt.plot(std_vec, FRR_vec, 'o-')
plt.axhline(y=spike_nonoise, color='r', linestyle='--', label='fire rate without noise')
plt.xlabel('Standard Deviation of Noise')
plt.ylabel('Firing Rate (Hz) at I_app = 4.000001e-9V')
plt.title('Sigma_V vs Firing Rate')
plt.grid(True)
plt.legend()
#plt.show()

#question 2 (c)
dt2 = dt / 10
t2 = np.arange(0, 2 + 0.1e-4, dt2)
I = np.zeros_like(t2)
V_m = np.zeros_like(t2)
V_m[0] = E_r
FRR3 = np.zeros_like(I_2)
FRR4 = np.zeros_like(I_2)

for i in range (0, 10):

    I[0:] = I_2[i]
    FRR3[i] = LIF_with_noise(V_m, I, t2, E_r, R_m, C_m, V_th, V_reset,0)[1] / 2
    FRR4[i] = LIF_with_noise(V_m, I, t2, E_r, R_m, C_m, V_th, V_reset,std_V2)[1] / 2
    # Calculate firing rate in Hz

#plotting question 2 (c)

plt.figure(figsize=(10, 6))
plt.plot(I_2, FRR, 'o-', label='dt = 0.1ms without noise')
plt.plot(I_2, FRR2, 'o-', label='dt = 0.1ms with noise std = ' + str(std_V2))
plt.plot(I_2, FRR3, 's-', label='dt = 0.01ms without noise')
plt.plot(I_2, FRR4, 'o-', label='dt = 0.01ms with noise std = ' + str(std_V2))

plt.xlabel('Applied Current (1e-9A)')
plt.ylabel('Firing Rate (Hz)')
plt.title('Firing-rate Curve Comparison : dt = 0.1ms vs dt = 0.01ms')
# Standard diviation of noise = 0.1
plt.legend()
plt.grid(True)

plt.show()
