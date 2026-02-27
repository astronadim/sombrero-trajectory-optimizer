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
    #ax3.plot(t, u_IspEff, label='Isp', linestyle=':', color='blue')
    
    ax3.set_xlabel('Time (years)')
    ax3.set_ylabel('Values')
    ax3.set_title('NN Inputs and Outputs Over Time')
    ax3.legend()
    ax3.set_ylim(-2, 2)
    ax3.grid(True)

    # Adding text annotations to the plot (only once, assuming similar information for both plots)
    info_text = (f'Generation: {current_generation} | individual: {individual_nr} (seed_value:{seed_value}\n\n'
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


    # Assuming generation, individual, and fitness_score are defined in your scope
    save_path = os.path.join(trajectory_figures_dir, f"gen{current_generation}-ind{individual_nr}-fitness{fitness_score}.png")
    plt.savefig(save_path)
    
    # Debug statements to verify the condition
    """
    print(f"show_trajectory is set to: {show_trajectory}")
    if show_trajectory:
        print("Displaying the trajectory plot.")
        plt.show()
    else:
        print("Not displaying the trajectory plot.")
    """
    plt.close(fig)



def update_and_save_fittest_solution(
                            fitness_score, timestamp, individual_nr, current_generation,
                            subFitnessScores, individual_chromosome, JGA_results, x, y, v_magnitude, Tx_h, Ty_h, Isp, t, delta_v, m_sc,
                            m_EPS, m_structure, m_payload, m_propellant, t_200AU, P_0, rtol, atol, n_hidden_layers, n_neuronsPerLayer,
                            activation_function, total_simulation_time_thrust_phase, mutation_rate, mutation_std_dev, u_IspEff,
                            u_PowerThrottle, theta_thrust, flight_path_angle, accumulated_true_anomaly_normalized, seed_value):
    
    R = np.sqrt(x**2 + y**2) / AU
    fittest_solution = pd.DataFrame({
        'timestamp': [timestamp],
        'seed_value': [seed_value],
        'individual_nr': [individual_nr],
        'current_generation': [current_generation],
        'fitness_score': [fitness_score],
        'subFitnessScores': [subFitnessScores],
        'individual_chromosome': [individual_chromosome],
        'JGA_results': [JGA_results],
        'm_sc': [m_sc],
        'm_EPS': [m_EPS],
        'm_structure': [m_structure],
        'm_payload': [m_payload],
        'm_propellant': [m_propellant],
        'delta_v': [delta_v],
        't_200AU': [t_200AU],
        'P_0': [P_0],
        'rtol': [rtol],
        'atol': [atol],
        'n_hidden_layers': [n_hidden_layers],
        'n_neuronsPerLayer': [n_neuronsPerLayer],
        'activation_function': [activation_function],
        'total_simulation_time_thrust_phase': [total_simulation_time_thrust_phase],
        'mutation_rate': [mutation_rate],
        'mutation_std_dev': [mutation_std_dev],
        'x': [x],
        'y': [y],
        'R': [R],
        'v_magnitude': [v_magnitude/29784],
        'flight_path_angle': [flight_path_angle],
        'accumulated_true_anomaly_normalized': [accumulated_true_anomaly_normalized],
        'Tx_h': [Tx_h],
        'Ty_h': [Ty_h],
        'Isp': [Isp],
        't': [t/(365*24*60*60)],
        'u_IspEff': [u_IspEff],
        'u_PowerThrottle': [u_PowerThrottle],
        'theta_thrust': [theta_thrust]
    })
    # Determine the directory where the script is located
    directory = os.path.dirname(os.path.abspath(__file__))

    # Construct the full file path by joining the directory with the filename
    full_path = os.path.join(directory, "fittest_solution.xlsx")

    # Save the fittest solution DataFrame to the Excel file
    fittest_solution.to_excel(full_path, index=False)



def save_fittest_individuals_per_generation(fittest_individuals_per_generation):
    # Constants
    AU = 1.496e11  # 1 Astronomical Unit in meters

    # List to collect data for DataFrame
    data_list = []

    for individual_data in fittest_individuals_per_generation:
        # Extract and process data
        x = individual_data['x']/AU
        y = individual_data['y']/AU
        v_magnitude = individual_data['v_magnitude']
        t = individual_data['t']
        # Calculate additional parameters
        R = np.sqrt(x**2 + y**2) / AU  # Distance from the sun in AU
        v_magnitude_normalized = v_magnitude / 29784  # Normalize velocity

        # Collect data into a dictionary
        data_entry = {
            'timestamp': individual_data['timestamp'],
            'seed_value': individual_data['seed_value'],
            'individual_nr': individual_data['individual_nr'],
            'current_generation': individual_data['current_generation'],
            'fitness_score': individual_data['fitness_score'],
            'subFitnessScores': individual_data['subFitnessScores'],
            'individual_chromosome': individual_data['individual_chromosome'],
            'JGA_results': individual_data['JGA_results'],
            'm_sc': individual_data['m_sc'],
            'm_EPS': individual_data['m_EPS'],
            'm_structure': individual_data['m_structure'],
            'm_payload': individual_data['m_payload'],
            'm_propellant': individual_data['m_propellant'],
            'delta_v': individual_data['delta_v'],
            't_200AU': individual_data['t_200AU'],
            'P_0': individual_data['P_0'],
            'rtol': individual_data['rtol'],
            'atol': individual_data['atol'],
            'n_hidden_layers': individual_data['n_hidden_layers'],
            'n_neuronsPerLayer': individual_data['n_neuronsPerLayer'],
            'activation_function': str(individual_data['activation_function']),
            'total_simulation_time_thrust_phase': individual_data['total_simulation_time_thrust_phase'],
            'mutation_rate': individual_data['mutation_rate'],
            'mutation_std_dev': individual_data['mutation_std_dev'],
            'x': x.tolist(),
            'y': y.tolist(),
            'R': R.tolist(),
            'v_magnitude': v_magnitude_normalized.tolist(),
            'flight_path_angle': individual_data['flight_path_angle'].tolist(),
            'accumulated_true_anomaly_normalized': individual_data['accumulated_true_anomaly_normalized'].tolist(),
            'Tx_h': individual_data['Tx_h'].tolist(),
            'Ty_h': individual_data['Ty_h'].tolist(),
            'Isp': individual_data['Isp'].tolist(),
            't': (t / (365 * 24 * 60 * 60)).tolist(),  # Convert time to years
            'u_IspEff': individual_data['u_IspEff'].tolist(),
            'u_PowerThrottle': individual_data['u_PowerThrottle'].tolist(),
            'theta_thrust': individual_data['theta_thrust'].tolist()
        }
        data_list.append(data_entry)

    # Create DataFrame
    fittest_individuals_df = pd.DataFrame(data_list)

    # Save to Excel file
    # Define the file path using timestamp from the first individual
    timestamp = fittest_individuals_per_generation[0]['timestamp']
    results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Results")
    file_name = f"fittest_individuals_per_generation-{timestamp}.xlsx"
    file_path = os.path.join(results_dir, file_name)

    # Save DataFrame to Excel
    fittest_individuals_df.to_excel(file_path, index=False)


def save_trajectories_and_fitness(all_trajectories_data):
    # Constants
    AU = 1.496e11  # 1 Astronomical Unit in meters

    # List to collect data for DataFrame
    data_list = []

    for data in all_trajectories_data:
        # Extract and process data
        x = data['x'] / AU
        y = data['y'] / AU
        v_magnitude = data['v_magnitude']
        t = data['t']
        # Calculate additional parameters
        R = np.sqrt(x**2 + y**2)  # Distance from the sun in AU
        v_magnitude_normalized = v_magnitude / 29784  # Normalize velocity (Earth's average orbital speed in m/s)
        time_in_years = t / (365 * 24 * 60 * 60)  # Convert time to years

        # Collect data into a dictionary
        data_entry = {
            'timestamp': data['timestamp'],
            'generation': data['generation'],
            'individual_nr': data['individual_nr'],
            'fitness_score': data['fitness_score'],
            'x': x.tolist(),
            'y': y.tolist(),
            'R': R.tolist(),
            'v_magnitude': v_magnitude_normalized.tolist(),
            't': time_in_years.tolist()
        }
        data_list.append(data_entry)

    # Create DataFrame
    trajectories_df = pd.DataFrame(data_list)

    # Save to Excel file
    # Define the file path using timestamp from the first trajectory
    timestamp = all_trajectories_data[0]['timestamp']
    results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Results")
    file_name = f"trajectories-{timestamp}.xlsx"
    file_path = os.path.join(results_dir, file_name)

    # Save DataFrame to Excel
    trajectories_df.to_excel(file_path, index=False)