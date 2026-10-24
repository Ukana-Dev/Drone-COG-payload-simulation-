#################################
#
#  Structure inspired by: lecturer's class example (AV overtaking cyclist)
#
#################################

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from IPython.display import HTML, display

# === CONSTANTS ===
G = 9.81                 # gravity [m/s^2]
I_DRONE = 0.02           # drone's own moment of inertia [kg*m^2]  
KP = 6.0                 # PID proportional gain [N*m/rad]
KI = 0.6                 # PID integral gain [N*m/(rad*s)]
KD = 1.2                 # PID derivative gain [N*m*s/rad]

PAYLOAD_MASS = 0.5       # [kg]  SHM Next node is ~0.5 kg
PAYLOAD_DISTANCE = 0.15  # [m]   distance of payload from the rotation axis

SIM_TIME = 10.0          # [s]
DT = 0.01                # [s] time step

TILT_LIMIT_DEG = 5.0     # [deg] max allowed tilt -> 
ARM_LENGTH = 0.25        # [m] half the drone width (only for drawing)
TILT_EXAGGERATION = 1    # drawing only: set to e.g. 5 to see small tilts better

#################################


# ==================================================================
#  SIMULATION FUNCTION - CORE, WRITTEN BY ME
# ==================================================================
def simulate(m=PAYLOAD_MASS, d=PAYLOAD_DISTANCE):
    """
    Simulates the tilt of the drone after a payload of mass m [kg]
    is attached at distance d [m]. The drone starts level and still.

    Must return a list called states, where each item is a dictionary:
        {'t': time [s], 'theta': tilt angle [rad], 'omega': angular speed [rad/s]}

    Steps to write yourself:
      0. compute I_total and M_gravity from m and d
      1. make a start state: theta, omega and the integral of theta all 0
      2. loop over time from 0 to SIM_TIME in steps of DT, and each step:
         a. controller torque from theta, omega and the integral
         b. angular acceleration from the torques and I_total
         c. update omega, theta and the integral (Euler: new = old + rate * DT)
         d. save the current values with states.append({...})
    """
    states = []
    #Heart of the code 

    M_gravity = m * G * d
    I_total   = I_DRONE + m * d**2
    s = {'theta': 0.0, 'omega': 0.0, 'integral': 0.0}

    for t in np.arange(0.0, SIM_TIME + DT / 2, DT):
        M_controller = KP*s['theta'] + KI*s['integral'] + KD*s['omega']
        alpha = (M_gravity - M_controller) / I_total
        s['omega']    = s['omega']    + alpha * DT
        s['theta']    = s['theta']    + s['omega'] * DT
        s['integral'] = s['integral'] + s['theta'] * DT
        states.append({'t': t, 'theta': s['theta'], 'omega': s['omega']})

    return states
# ==================================================================


#################################
#
#  Structure inspired by: lecturer's class example (AV overtaking cyclist)
#
#################################

# === RUN SIMULATION ===
states = simulate()
if len(states) == 0:
    raise ValueError("simulate() returned no states yet - write your time loop first!")

# === ANIMATION SETUP ===
fig, (ax_drone, ax_plot) = plt.subplots(1, 2, figsize=(11, 4))

# Left: drone seen from the side (payload on the right)
arm_line, = ax_drone.plot([], [], 'b-', lw=4)
body, = ax_drone.plot([], [], 'ks', ms=12)
payload_dot, = ax_drone.plot([], [], 'o', color='orange', ms=12)
ax_drone.set_xlim(-0.35, 0.35)
ax_drone.set_ylim(-0.2, 0.2)
ax_drone.set_aspect('equal')
ax_drone.grid(True)
ax_drone.set_xlabel("X (m)")
ax_drone.set_ylabel("Y (m)")

# Right: tilt angle over time
times = [s['t'] for s in states]
tilts = [np.degrees(s['theta']) for s in states]
ax_plot.plot(times, tilts, 'b-')
ax_plot.axhline(TILT_LIMIT_DEG, color='r', ls='--', label=f"{TILT_LIMIT_DEG}° limit")
ax_plot.axhline(-TILT_LIMIT_DEG, color='r', ls='--')
marker, = ax_plot.plot([], [], 'ro')
ax_plot.set_xlabel("Time (s)")
ax_plot.set_ylabel("Tilt angle (deg)")
ax_plot.set_title(f"m = {PAYLOAD_MASS} kg, d = {PAYLOAD_DISTANCE} m")
ax_plot.grid(True)
ax_plot.legend()


def rotate(x, y, angle):
    # rotate point (x, y) around the origin by angle [rad]
    return (x * np.cos(angle) - y * np.sin(angle),
            x * np.sin(angle) + y * np.cos(angle))


def update(i):
    s = states[i]
    angle = -s['theta'] * TILT_EXAGGERATION   # payload side (right) dips down

    x1, y1 = rotate(-ARM_LENGTH, 0, angle)
    x2, y2 = rotate(ARM_LENGTH, 0, angle)
    arm_line.set_data([x1, x2], [y1, y2])
    body.set_data([0], [0])

    px, py = rotate(PAYLOAD_DISTANCE, -0.04, angle)   # payload hangs just below arm
    payload_dot.set_data([px], [py])

    marker.set_data([s['t']], [np.degrees(s['theta'])])
    ax_drone.set_title(f"t = {s['t']:.2f} s   tilt = {np.degrees(s['theta']):.2f}°")
    return arm_line, body, payload_dot, marker


STEP = 5   # show every 5th state to keep the animation light
anim = animation.FuncAnimation(
    fig, update, frames=range(0, len(states), STEP), interval=DT * STEP * 1000, blit=False
)

plt.close(fig)  # prevent static fig in Colab

# === DISPLAY IN COLAB ===
display(HTML(anim.to_jshtml()))

# === COMPARE DIFFERENT PAYLOAD MASSES (static plot) ===
plt.figure(figsize=(7, 4))
for m in [0.1, 0.3, 0.5, 0.8]:
    run = simulate(m=m)
    plt.plot([s['t'] for s in run], [np.degrees(s['theta']) for s in run], label=f"m = {m} kg")
plt.axhline(TILT_LIMIT_DEG, color='r', ls='--', label=f"{TILT_LIMIT_DEG}° limit")
plt.xlabel("Time (s)")
plt.ylabel("Tilt angle (deg)")
plt.title(f"Effect of payload mass (d = {PAYLOAD_DISTANCE} m)")
plt.grid(True)
plt.legend()
plt.show()


# from matplotlib.animation import PillowWriter
# anim.save('drone_tilt.gif', writer=PillowWriter(fps=int(1 / (DT * STEP))))

