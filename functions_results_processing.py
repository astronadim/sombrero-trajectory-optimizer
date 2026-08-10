import os
import datetime
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.lines import Line2D
matplotlib.use('Agg')
import numpy as np
import pandas as pd

AU = 1.496e11

def plot_and_save_trajectory(x, y, v_magnitude, Tx_h, Ty_h, Isp, JGA_results, individual_chromosome, 
                             current_generation, individual_nr, fitness_score, subFitnessScores, 
                             m_sc, m_EPS, m_structure, m_payload, m_propellant, delta_v, t_200AU, P_0, u_IspEff, 
                             u_PowerThrottle, theta_thrust, flight_path_angle, accumulated_true_anomaly_normalized, t, seed_value, trajectory_figures_dir):
    # Convert positions to AU
    
    X, Y = x / AU, y / AU
    R = np.sqrt(X**2 + Y**2)
    t = t / (365.25 * 24 * 3600)
    V_magnitude = v_magnitude / 29784
    
    # Create the figure and axes for three subplots
    fig = plt.figure(figsize=(20, 15))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1])
    
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, :])
    
    def plot_trajectory(ax, limit_value):
        ax.set_aspect('equal')
        sc = ax.scatter(X, Y, c=v_magnitude, cmap='jet', label='Spacecraft Trajectory', s=10)
        ax.plot(0, 0, 'yo', markersize=10, label='Sun')
        ax.plot(1, 0, 'bo', markersize=5, label='Earth at launch')

        # Draw orbits for Earth and Jupiter
        earth_orbit = plt.Circle((0, 0), 1, color='blue', fill=False, linestyle='dashed')
        jupiter_orbit = plt.Circle((0, 0), 5.2, color='orange', fill=False, linestyle='dashed')
        ax.add_artist(earth_orbit)
        ax.add_artist(jupiter_orbit)

        # Create custom legend handles
        custom_lines = [
            Line2D([0], [0], color='blue', linestyle='dashed', label='Earth Orbit'),
            Line2D([0], [0], color='orange', linestyle='dashed', label='Jupiter Orbit')
        ]

        ax.set_xlabel('x position (AU)')
        ax.set_ylabel('y position (AU)')
        ax.set_title('Trajectory of Spacecraft')
        ax.axis([-limit_value, limit_value, -limit_value, limit_value])
        ax.grid(True)
        ax.legend(handles=custom_lines + ax.get_legend_handles_labels()[0])

        cbar = plt.colorbar(sc, ax=ax)
        cbar.set_label('Heliocentric Velocity (m/s)')

        # Find the index of the maximum thrust
        max_length_index = np.argmax(np.sqrt(Tx_h**2 + Ty_h**2))
        max_thrust = np.sqrt(Tx_h[max_length_index]**2 + Ty_h[max_length_index]**2)

        # Scale the length of the maximum thrust vector
        scale_factor = 0.5 / max_thrust

        # Add text with maximum thrust at the position of the arrow with maximum thrust
        ax.text(X[max_length_index] - 1, Y[max_length_index] - 1, f'T_max: {max_thrust:.1e} N', color='black', fontsize=6)

        # Add text to JGA
        ax.text(JGA_results['x_JGA'].values[0] - 0.5, JGA_results['y_JGA'].values[0],
                f'JGA_delta_v: {JGA_results["JGA_delta_v"].values[0]:.1e} m/s', color='black', fontsize=6)
        
        n_vectors = min(100, len(X))
        indices = np.linspace(0, len(X)-1, n_vectors, dtype=int)
        for i in indices:
            ax.quiver(X[i], Y[i], Tx_h[i] * scale_factor, Ty_h[i] * scale_factor, color='black', angles='xy', scale_units='xy', scale=1, width=0.002)

    # Plot the original trajectory
    plot_trajectory(ax1, limit_value=6)
    
    # Plot the trajectory with limited axis values
    plot_trajectory(ax2, limit_value=2)

    # Plot the third subplot with the specified data
    ax3.plot(t, R, label='R', linestyle='-', color='red')
    ax3.plot(t, V_magnitude, label='V_magnitude', linestyle='--', color='red')
    ax3.plot(t, accumulated_true_anomaly_normalized, label='Accumulated True Anomaly Normalized', linestyle='-.', color='red', alpha=0.7)
    ax3.plot(t, flight_path_angle, label='Flight Path Angle', linestyle=':', color='red')

    ax3.plot(t, theta_thrust, label='Theta Thrust', linestyle='--', color='blue')
    ax3.plot(t, u_PowerThrottle, label='Power Throttle', linestyle='-.', color='blue')
    
    ax3.set_xlabel('Time (years)')
    ax3.set_ylabel('Values')
    ax3.set_title('NN Inputs and Outputs Over Time')
    ax3.legend()
    ax3.set_ylim(-2, 2)
    ax3.grid(True)

    # Adding text annotations to the plot (only once, assuming similar information for both plots)
    info_text = (f'Generation: {current_generation} | individual: {individual_nr} (seed_value: {seed_value})\n\n'
                 f'Initial Conditions\n'
                 f'C3: {individual_chromosome[0]:.0f} km²/s²\n'
                 f'Launch Angle: {np.degrees(individual_chromosome[1]):.2f}°\n'
                 f'Initial Payload/Propellant Ratio: {individual_chromosome[2]:.2f}\n'
                 f'Initial Spacecraft Mass: {m_sc[0]:.0f} kg\n'
                 f'P_0: {P_0:.0f} W\n\n'

                 f'Final values\n'
                 f'Time to 200 AU: {t_200AU / (365.25 * 24 * 3600):.2f} years\n'
                 f'Delta V: {delta_v[-1]:.0f} m/s\n'
                 f'Final Spacecraft Mass: {m_sc[-1]:.0f} kg\n'
                 f'Payload Mass: {m_payload:.0f} kg\n'
                 f'EPS Mass: {m_EPS:.0f} kg\n'
                 f'Structure Mass: {m_structure:.0f} kg\n'
                 f'Propellant Mass: {m_propellant:.0f} kg\n\n'
                                 
                 f'Fitness Score: {fitness_score}\n'
                 f'Subfitness Scores:\n'
                 f'  Solar Oberth: {subFitnessScores["Solar Oberth"].values[0]}\n'
                 f'  Approach Jupiter: {subFitnessScores["Approach Jupiter"].values[0]}\n'
                 f'  Solar Escape: {subFitnessScores["Solar Escape"].values[0]}\n'
                 f'  Time to 200 AU: {subFitnessScores["Time to 200 AU"].values[0]}\n'
                 f'  Payload: {subFitnessScores["Payload"].values[0]}')

    plt.gcf().text(0.02, 0.5, info_text, fontsize=8, verticalalignment='center', bbox=dict(facecolor='white', alpha=0.5))


    save_path = os.path.join(trajectory_figures_dir, f"gen{current_generation}-ind{individual_nr}-fitness{fitness_score}.png")
    plt.savefig(save_path)
    
    plt.close(fig)




