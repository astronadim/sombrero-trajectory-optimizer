import numpy as np
import pandas as pd
from functions_evolutionary import load_parameters_into_nn
from scipy.integrate import solve_ivp

## Constants ##
G = 6.67430e-11  # gravitational constant in m^3 kg^-1 s^-2
GM_sun = 1.327e20  # gravitational parameter of the Sun in m^3/s^2
AU = 1.496e11  # 1 Astronomical Unit in meters
v_Earth = 29.78e3
g0 = 9.81  # m/s^2
GM_jupiter = 1.26686534e17  # m^3/s^2
radius_jupiter = 71492e3  # m (Jupiter's mean radius)
YEAR = 365.25 * 24 * 3600  # Seconds in a year

## define spacecraft parameters ##
mu_structure = 0.3  # structural mass ratio
alpha_SEP = 200  # specific power of Solar Electric Propulsion at 1 AU [W/kg]
P_throttle_crit_ratio = 0.01  # if chi_throttle_NN < P_throttle_crit, then P_throttle = 0
JGA_Periapsis = 1.34 * radius_jupiter  # periapsis distance for Jupiter Gravity Assist

# Default electric propulsion parameters (can be overridden via simulate_trajectory)
global_Isp_config = 6000.0
global_efficiency_config = 0.75

# Define the equations of motion
def equations_of_motion(t, state, m_sc_0, m_EPS):
    global accumulated_true_anomaly_normalized, alreadyPerformedRevolution

    x, y, vx, vy, m_sc, delta_v = state

    # determine inputs for neurocontroller from state
    R, true_anomaly, flight_path_angle, V_magnitude, mu_PropPay = transform_state(state, m_sc_0, m_EPS)

    # calculation of accumulated true anomaly needs special attention for the case of a revolution around the sun. In this case, the true anomaly is reset to 0, but the accumulated true anomaly should be increased by 2pi
    if 0.0 < true_anomaly/(2*np.pi) < 0.05 and 0.95 < accumulated_true_anomaly_normalized < 1 and not alreadyPerformedRevolution:
        alreadyPerformedRevolution = True

    if alreadyPerformedRevolution == True:
        accumulated_true_anomaly = true_anomaly + 2*np.pi
    else:
        accumulated_true_anomaly = true_anomaly

    accumulated_true_anomaly_normalized = accumulated_true_anomaly/(2*np.pi)

    # prevents a bug
    if accumulated_true_anomaly_normalized > 1.6 and R < 3:
        accumulated_true_anomaly_normalized = accumulated_true_anomaly_normalized -1
  
    # determine thrust angle and power throttle from neurocontroller
    theta_thrust, u_PowerThrottle = NeuroController(R, accumulated_true_anomaly_normalized, flight_path_angle, V_magnitude)

    
    theta_thrust = np.pi * theta_thrust  # convert from [-1, 1] to [-pi, pi] thrust angles behind -90 and 90 degrees must be possible for initial perihelion decrease!!   
    
    u_PowerThrottle_Lower, u_PowerThrottle_Upper = 0, 0.8
    P_available = P_0 * (1 / R) ** 1.5

    if  P_available < P_throttle_crit_ratio * P_max or u_PowerThrottle < u_PowerThrottle_Lower:
        u_PowerThrottle = 0
    elif u_PowerThrottle_Lower <= u_PowerThrottle <= u_PowerThrottle_Upper:
        u_PowerThrottle = (u_PowerThrottle - u_PowerThrottle_Lower) / (u_PowerThrottle_Upper - u_PowerThrottle_Lower)
    else:
        u_PowerThrottle = 1

    """
    if P_available > 81540: # max power for 6 RIT-22: 6*13.59 kW
        P_available = 81540
    """
    
    u_IspEff = 1 # u_IspEff can be used to throttle Isp. For now, it is set to 1, meaning full Isp is used, to facilitate the training of the neural network. in a future version, NN can be augmented by including this parameter
    m_PropPay = (m_sc - mu_structure * m_sc_0 - m_EPS)  # subtract structural and EPS spacecraft mass to arrive at payload and propellant
    Tx_sc, Ty_sc, Isp = determine_spacecraft_centric_thrust_components(
        theta_thrust, u_IspEff, u_PowerThrottle, P_available, R, m_PropPay, mu_PropPay
    )
    
    Tx_h, Ty_h = convert_spacecraft_centric_to_heliocentric(Tx_sc, Ty_sc, vx, vy)

    # Save to the global lists
    global_Tx_h.append(Tx_h); global_Ty_h.append(Ty_h); global_Isp.append(Isp); global_theta_thrust.append(theta_thrust); global_u_IspEff.append(u_IspEff); global_u_PowerThrottle.append(u_PowerThrottle); global_t.append(t); global_flight_path_angle.append(flight_path_angle); global_accumulated_true_anomaly_normalized.append(accumulated_true_anomaly_normalized)

    
    r = np.sqrt(x**2 + y**2)
    ax = -GM_sun * x / r**3 + Tx_h / m_sc
    ay = -GM_sun * y / r**3 + Ty_h / m_sc
    mdot_current = -np.linalg.norm([Tx_h, Ty_h]) / (Isp * g0)
    a_thrust = np.linalg.norm([Tx_h, Ty_h]) / m_sc

    # print out results
    #print(f"t:{format_time(t)}, mu_PP:{mu_PropPay:.2f}, R:{R:.2e}, true_a:{(true_anomaly/np.pi*180):.2f}, th_vR:{(flight_path_angle/np.pi*180):.2f}, V:{V_magnitude:.2f}, th_T:{(theta_thrust/np.pi*180):.2f}, u_Isp:{u_IspEff:.2f}, u_PT:{u_PowerThrottle:.2f}, P_av:{P_available:.2e}, Txh:{Tx_h:.2e}, Tyh:{Ty_h:.2e}, Isp:{Isp:.0f}")

    return [vx, vy, ax, ay, mdot_current, a_thrust]

# transform state to neural network input representation
def transform_state(state, m_sc_0, m_EPS):
    x, y, vx, vy, m_sc, _ = state
    V_vector_normalized = np.array([vx / v_Earth, vy / v_Earth])
    location_vector_normalized = np.array([x / AU, y / AU])
    R = np.sqrt(x**2 + y**2) / AU  # radial direction from Earth in AU

    true_anomaly = np.arctan2(y, x) # between 0 and 2pi:  
    if true_anomaly < 0:
        true_anomaly += 2 * np.pi
    flight_path_angle = angle_between_vectors(V_vector_normalized, location_vector_normalized)
    V_magnitude = np.linalg.norm(V_vector_normalized)
    mu_PropPay = (
        m_sc - mu_structure * m_sc_0 - m_EPS
    ) / m_sc_0  # subtract structural and EPS spacecraft mass to arrive at payload and propellant
    return R, true_anomaly, flight_path_angle, V_magnitude, mu_PropPay

# determine thrust components and Isp from neurocontrol inputs
def determine_spacecraft_centric_thrust_components(theta_thrust, u_IspEff, u_PowerThrottle, P_available, R, m_PropPay, mu_PropPay):
    global global_Isp_config, global_efficiency_config
    Isp = global_Isp_config  # Use configured Isp
    efficiency = global_efficiency_config  # Use configured efficiency

    T_overall = u_PowerThrottle * 2 * P_available * efficiency / (g0 * Isp)  # overall thrust

    Tx_sc = -np.sin(theta_thrust) * T_overall  # thrust in x direction
    Ty_sc = np.cos(theta_thrust) * T_overall  # thrust in y direction

    if R > 5.2 or m_PropPay < 250 or mu_PropPay < 0.1:  # don't thrust anymore if beyond Jupiter or payload mass or mass ratio below threshold
        Tx_sc = 0
        Ty_sc = 0

    return Tx_sc, Ty_sc, Isp

def convert_spacecraft_centric_to_heliocentric(Tx_sc, Ty_sc, vx, vy):

    # determine angle for conversion of spacecraft centric to heliocentric thrust components
    angle_spacecraftcentric_to_heliocentric = np.arctan2(vy, vx) # between 0 and 2pi  
    if angle_spacecraftcentric_to_heliocentric < 0:
        angle_spacecraftcentric_to_heliocentric += 2 * np.pi

    angle_spacecraftcentric_to_heliocentric = angle_spacecraftcentric_to_heliocentric - np.pi/2  


    # Rotation matrix for converting from spacecraft-centric to heliocentric coordinates
    rotation_matrix = np.array([[np.cos(angle_spacecraftcentric_to_heliocentric), -np.sin(angle_spacecraftcentric_to_heliocentric)], [np.sin(angle_spacecraftcentric_to_heliocentric), np.cos(angle_spacecraftcentric_to_heliocentric)]])

    # Spacecraft-centric thrust vector
    thrust_sc = np.array([Tx_sc, Ty_sc])

    # Applying the rotation matrix to convert to heliocentric coordinates
    thrust_heliocentric = np.dot(rotation_matrix, thrust_sc)

    # Extracting the heliocentric thrust components
    Tx_h, Ty_h = thrust_heliocentric

    return Tx_h, Ty_h

def angle_between_vectors(a, b):
    # Ensure a and b are numpy arrays
    a = np.array(a)
    b = np.array(b)

    # Normalize the vectors
    a_norm = a / np.linalg.norm(a)
    b_norm = b / np.linalg.norm(b)

    # Calculate the dot product and the determinant
    dot_product = np.dot(a_norm, b_norm)
    determinant = np.linalg.det(np.array([a_norm, b_norm]))

    # Calculate the angle in radians
    angle = np.arctan2(determinant, dot_product)

    return angle

def format_time(seconds):
    years = seconds // (365.25 * 24 * 3600)
    seconds %= 365.25 * 24 * 3600
    months = seconds // (30.44 * 24 * 3600)
    seconds %= 30.44 * 24 * 3600
    days = seconds // (24 * 3600)
    seconds %= 24 * 3600
    hours = seconds // 3600

    return f"{int(years)}y{int(months)}m{int(days)}d{int(hours)}h"

# Event function to stop integration when r > 5.2 AU
def terminal_events_r_exceeds_5_2_AU_or_undercuts_0_15_AU(t, state, *args):
    x, y, vx, vy, m_sc, delta_v = state
    r = np.sqrt(x**2 + y**2)

    return max(r - 5.2 * AU, 0.15 * AU - r)

def PerformJupiterGravityAssist(state, JGA_Periapsis, enable_jga=True):
    """Perform Jupiter Gravity Assist.
    
    Args:
        state: Current spacecraft state [x, y, vx, vy, m_sc, delta_v]
        JGA_Periapsis: Periapsis distance for JGA
        enable_jga: If True, perform velocity rotation. If False, pass through unchanged.
    
    Returns:
        Updated state and JGA_results DataFrame
    """
    x, y, vx, vy, m_sc, delta_v = state

    v_spacecraft = np.array([vx, vy])
    pos_vec = np.array([x, y])
    
    if enable_jga:
        print(f"Performing JGA at x={x/AU:.2f} AU, y={y/AU:.2f} AU | Before JGA: v_magnitude = {np.linalg.norm(v_spacecraft):.2f} m/s")
    else:
        print(f"At Jupiter encounter (JGA disabled) at x={x/AU:.2f} AU, y={y/AU:.2f} AU | v_magnitude = {np.linalg.norm(v_spacecraft):.2f} m/s")

    if enable_jga:
        # Perform the gravity assist rotation
        v_jupiter = np.array([-y, x]) / np.linalg.norm(pos_vec) * 13060  # Jupiter's velocity magnitude

        v_relative = v_spacecraft - v_jupiter
        turn_angle = 2 * np.arcsin(1 / (1 + JGA_Periapsis * np.linalg.norm(v_relative)**2 / GM_jupiter))

        rotation_matrix = np.array([[np.cos(turn_angle), -np.sin(turn_angle)], [np.sin(turn_angle), np.cos(turn_angle)]])
        v_final = np.dot(rotation_matrix, v_relative) + v_jupiter

        print(f"  After JGA rotation: v_magnitude = {np.linalg.norm(v_final):.2f} m/s, delta_v = {np.linalg.norm(v_final) - np.linalg.norm(v_spacecraft):.2f} m/s")
        
        state[2:4] = v_final
        jga_delta_v = np.linalg.norm(v_final) - np.linalg.norm(v_spacecraft)
        energy_gain_ratio = 0.5 * (np.linalg.norm(v_final)**2 - np.linalg.norm(v_spacecraft)**2) / 5.5e9
    else:
        # JGA disabled: no velocity change
        v_final = v_spacecraft
        jga_delta_v = 0.0
        energy_gain_ratio = 1.0
        print(f"  No gravity assist performed (enable_jga=False)")

    JGA_results = pd.DataFrame({
        'x_JGA': [x/AU], 
        'y_JGA': [y/AU], 
        'JGA_delta_v': [float(jga_delta_v) if np.isfinite(jga_delta_v) else 0.0], 
        'v_mag_afterJGA': [float(np.linalg.norm(v_final)) if np.isfinite(np.linalg.norm(v_final)) else 0.0],
        'energy_gain_ratio': [float(energy_gain_ratio) if np.isfinite(energy_gain_ratio) else 0.0]
    })

    return state, JGA_results

def simulate_trajectory(individual_chromosone, fitness_settings, n_hidden_layers, n_neuronsPerLayer, activation_function, total_simulation_time_thrust_phase, rtol, atol, enable_jga=True, Isp_config=6000.0, efficiency_config=0.75):

    # Store EP parameters as globals for use in equations_of_motion
    global global_Isp_config, global_efficiency_config
    global_Isp_config = Isp_config
    global_efficiency_config = efficiency_config

    ## extract parameters from chromosone ##
    # initial conditions
    C3, launch_angle, mu_PayloadPropellantInitial = individual_chromosone[0], individual_chromosone[1], individual_chromosone[2]
    mu_PropPay = mu_PayloadPropellantInitial
    if C3<0 or C3>150: # criteria for letting an individual be reinitialized. this condition might be the consequence of a mutation
        C3 = 0
    
    # neural network parameters: 1) initialise NeuroController, 2) load parameters
    global NeuroController  # global to be able to use it in equations_of_motion
    #NeuroController = initialize_neural_network(n_hidden_layers=n_hidden_layers, n_neuronsPerLayer=n_neuronsPerLayer, activation_function=activation_function)
    NeuroController = load_parameters_into_nn(individual_chromosone, n_hidden_layers, n_neuronsPerLayer, activation_function)

    
    ## initial state ##
    delta_v = 0
    m_sc_0 = 15189 * np.exp (-0.02165 * C3) # 1520#45000 * np.exp(-0.02012 * C3)  # assumption: Launch with SLS Block 2
    vx0 = -np.sin(launch_angle) * np.sqrt(C3 * 10**6)  # initial velocity in x direction in m/s (Earth's orbital velocity around the Sun). minus because launch_angle is always negative.
    vy0 =  np.cos(launch_angle) * np.sqrt(C3 * 10**6) + v_Earth  # initial velocity in y direction
    state0 = np.array([1 * AU, 0, vx0, vy0, m_sc_0, delta_v])  # Updated to remove Tx_h, Ty_h, and Isp

    m_EPS = (1 - mu_structure - mu_PayloadPropellantInitial) * m_sc_0  # initial Electrical Power System mass. Assumption: structural spacecraft mass is 20% of initial overall mass
    global P_0, P_max  # global variable to be able to use it in equations_of_motion
    P_0 = m_EPS * alpha_SEP  # initial power follows from m_EPS and specific power of the Solar Electric Propulsion system
    P_max = P_0 * (1 / fitness_settings['R_SolarOBerth_Design']) ** 1.5 # to do: change 0.3 to R_SolarOberth_Design


    # global true anomaly
    global alreadyPerformedRevolution, accumulated_true_anomaly_normalized
    alreadyPerformedRevolution = False
    accumulated_true_anomaly_normalized = 0

    # Clear global lists
    global global_Tx_h, global_Ty_h, global_Isp, global_t, global_u_IspEff, global_u_PowerThrottle, global_theta_thrust, global_flight_path_angle, global_accumulated_true_anomaly_normalized
    global_Tx_h, global_Ty_h, global_Isp, global_t, global_u_IspEff, global_u_PowerThrottle, global_theta_thrust, global_flight_path_angle, global_accumulated_true_anomaly_normalized = [], [], [], [], [], [], [], [], []
  

    print(f"Initial conditions - C3: {C3:.2f}, m_sc_0:{m_sc_0} launch_angle: {launch_angle/np.pi*180:.2f}, mu_PayloadPropellantInitial: {mu_PayloadPropellantInitial:.2f}, mu_EPS: {(m_EPS/m_sc_0):.2f}")

    # initialisation
    current_time = 0
    JGA_results = pd.DataFrame({'x_JGA': [np.nan], 'y_JGA': [np.nan], 'JGA_delta_v': [np.nan], "v_mag_afterJGA":[np.linalg.norm([np.nan])], 'energy_gain_ratio': [np.nan]})
    t_200AU = np.nan

    terminal_events_r_exceeds_5_2_AU_or_undercuts_0_15_AU.terminal = True
    terminal_events_r_exceeds_5_2_AU_or_undercuts_0_15_AU.direction = 1

    t_span = [0, total_simulation_time_thrust_phase]  # One year in seconds

    # Call solve_ivp to integrate the equations of motion with up until the JGA at 5.2 AU or total_simulation_time_thrust_phase
    sol = solve_ivp(equations_of_motion, t_span, state0, args=(m_sc_0, m_EPS), events=terminal_events_r_exceeds_5_2_AU_or_undercuts_0_15_AU, rtol=rtol, atol=atol, dense_output=True)

    # Extract results
    t = sol.t
    x, y, vx, vy, m_sc, delta_v = sol.y

    # Check if the JGA event triggered and handle the gravity assist and ensuing simulation of trajectory for plot
    if sol.status == 1 and np.sqrt(x[-1]**2 + y[-1]**2) > 5.2*AU:  # 1 indicates a JGA event was triggered; > 5.2*AU condition is further proof for JGA
    
        state = [x[-1], y[-1], vx[-1], vy[-1], m_sc[-1], delta_v[-1]]
        state, JGA_results = PerformJupiterGravityAssist(state, JGA_Periapsis, enable_jga=enable_jga)

        # if escape velocity reached, determine time to 200 AU
        if JGA_results['v_mag_afterJGA'][0] > np.sqrt(2 * GM_sun / np.sqrt(x[-1]**2 + y[-1]**2)):
            current_time = t[-1]
            t_200AU = determine_t_200AU(state, current_time, rtol, atol)
            print(f"Time to reach 200 AU: {t_200AU / (365.25 * 24 * 3600)} years")

        # Continue integration after gravity assist, only for the plot
        t_span_new = [t[-1], t[-1]+1*30*24*60*60]  # Continue for the remaining simulation time
        sol = solve_ivp(equations_of_motion, t_span_new, state, args=(m_sc_0, m_EPS), rtol=rtol, atol=atol, dense_output=True)
        t = np.concatenate((t, sol.t))
        x, y, vx, vy, m_sc, delta_v = [np.concatenate((arr, sol.y[idx])) for idx, arr in enumerate([x, y, vx, vy, m_sc, delta_v])]


    # Extract additional results
    v_magnitude = np.sqrt(vx**2 + vy**2)
    m_propellant = m_sc[0] - m_sc[-1] # difference of initial and final spacecraft mass is propellant
    m_structure  = mu_structure*m_sc[0]
    m_payload    = m_sc[0] - m_structure - m_EPS - m_propellant

    #print(f"Final delta v = {delta_v[-1]} m/s | m_sc_0 = {m_sc_0:.0f} kg | m_payload = {m_payload:.0f} kg | m_propellant = {m_propellant:.0f} kg | m_structure = {(m_structure):.0f} kg, m_EPS = {m_EPS:.0f} kg,  ")
    if enable_jga or pd.notna(JGA_results['JGA_delta_v'].values[0]):
        print(f"JGA results: JGA delta_v = {JGA_results['JGA_delta_v'].values[0]:.2f} m/s | v_sc after JGA = {JGA_results['v_mag_afterJGA'].values[0]:.2f} m/s  | energy gain ratio = {JGA_results['energy_gain_ratio'].values[0]:.5f}")
    if np.isfinite(t_200AU):  # Fixed: was t_200AU != np.nan which is always True
        print(f"t200AU: {t_200AU/YEAR} years")

    Tx_h, Ty_h, Isp, global_t, global_u_IspEff, global_u_PowerThrottle, global_theta_thrust, global_flight_path_angle, global_accumulated_true_anomaly_normalized = np.array(global_Tx_h), np.array(global_Ty_h), np.array(global_Isp), np.array(global_t), np.array(global_u_IspEff), np.array(global_u_PowerThrottle), np.array(global_theta_thrust), np.array(global_flight_path_angle), np.array(global_accumulated_true_anomaly_normalized)
    
    

    indices = [np.where(global_t == t_val)[0][0] for t_val in t if t_val in global_t] # if t_val in global_t is needed in case an event e.g. R<0.15AU is triggered. in this case t values might exist, that dont occur in global_t. therefore only those before the event shall be selected
    # make sure x, y, v_magnitude have same length as t, by selecting only the first entries of these vectors. this means cutting away the last entry that corresponds to after an event occured
    x, y, v_magnitude, t = x[0:len(indices)], y[0:len(indices)], v_magnitude[0:len(indices)], t[0:len(indices)]

    Tx_h, Ty_h, Isp, u_IspEff, u_PowerThrottle, theta_thrust, flight_path_angle, accumulated_true_anomaly_normalized = Tx_h[indices], Ty_h[indices], Isp[indices], global_u_IspEff[indices], global_u_PowerThrottle[indices], global_theta_thrust[indices], global_flight_path_angle[indices], global_accumulated_true_anomaly_normalized[indices]
    



    return x, y, v_magnitude, Tx_h, Ty_h, Isp , t, delta_v, m_sc, m_EPS, m_structure, m_payload, m_propellant, JGA_results, t_200AU, P_0, u_IspEff, u_PowerThrottle, theta_thrust, flight_path_angle, accumulated_true_anomaly_normalized


def determine_t_200AU(state, current_time, rtol, atol):
    """Determine time to reach 200 AU.
    
    Args:
        state: Current spacecraft state [x, y, vx, vy, m_sc, delta_v]
        current_time: Current simulation time in seconds
        rtol, atol: Integration tolerances
    
    Returns:
        Time to reach 200 AU in seconds, or np.inf if not reachable
    """
    x, y, vx, vy, m_sc, delta_v = state
    
    max_time = 100 * YEAR  # 100 years in seconds

    def equations(t, state_reduced):
        x, y, vx, vy = state_reduced
        r = np.sqrt(x**2 + y**2)
        ax = -GM_sun * x / r**3
        ay = -GM_sun * y / r**3
        return [vx, vy, ax, ay]

    def event_200AU_or_undercut_2_AU(t, state_reduced):
        x, y, vx, vy = state_reduced
        r = np.sqrt(x**2 + y**2)
        return max(r - 200 * AU, 2 * AU - r)

    event_200AU_or_undercut_2_AU.terminal = True
    event_200AU_or_undercut_2_AU.direction = 1

    # Initialize sol to None before the velocity check
    sol = None
    
    # Check if velocity larger than escape velocity
    v_mag = np.sqrt(vx**2 + vy**2)
    r_current = np.sqrt(x**2 + y**2)
    v_escape = np.sqrt(2 * GM_sun / r_current)
    
    if v_mag > v_escape:
        state_reduced0 = [x, y, vx, vy]
        sol = solve_ivp(equations, [0, max_time], state_reduced0, events=event_200AU_or_undercut_2_AU, rtol=rtol, atol=atol)

    # Only access sol if it is not None
    if sol is not None and sol.t_events[0].size > 0:
        t_200AU = (current_time + sol.t_events[0][0])
        print(f"t200AU: {t_200AU/YEAR} years")
        return t_200AU
    
    else:
        return np.inf  # If the spacecraft does not reach 200 AU within the maximum time
