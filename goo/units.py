"""SI throughout (SPEC §2). Multiply to write a value, divide to read one: 25 * um, h / um."""
m, mm, um, nm = 1.0, 1e-3, 1e-6, 1e-9
s, ms, minute, hour, day = 1.0, 1e-3, 60.0, 3600.0, 86400.0
kg, g, mg = 1.0, 1e-3, 1e-6
J, mJ = 1.0, 1e-3
N, mN = 1.0, 1e-3                 # surface tension: mN / m
Pa, kPa = 1.0, 1e3
Pa_s, mPa_s = 1.0, 1e-3
K = 1.0
litre, mL, uL = 1e-3, 1e-6, 1e-9  # m^3

k_B = 1.380649e-23                # J/K, exact
N_A = 6.02214076e23               # 1/mol, exact
R_gas = k_B * N_A                 # J/(mol K)
c_0 = 299792458.0                 # m/s, exact
g_0 = 9.80665                     # m/s^2, standard gravity
T_room = 293.15                   # K, a convention (20 C)


def celsius(t):
    """Kelvin from degrees Celsius."""
    return t + 273.15
