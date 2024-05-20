import math
import simplekml
import sys, os, csv
parent_directory = os.path.split(os.getcwd())[0]
cur_directory = os.path.join(parent_directory,os.path.split(os.getcwd())[1])
ephemeris_data_directory = os.path.join(parent_directory, 'data')
sys.path.insert(0, parent_directory)
from datetime import datetime, timezone, timedelta
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import navpy
from gnssutils import EphemerisManager

#input_filepath = os.path.join(cur_directory, 'data', 'sample', 'gnss_log_2024_04_13_19_53_33.txt')
input_filepath = input("Enter the path to the input file: ")

def calculate_satellite_position(ephemeris, transmit_time):
    mu = 3.986005e14
    OmegaDot_e = 7.2921151467e-5
    F = -4.442807633e-10
    sv_position = pd.DataFrame()
    sv_position['sv'] = ephemeris.index
    sv_position.set_index('sv', inplace=True)
    sv_position['t_k'] = transmit_time - ephemeris['t_oe']
    A = ephemeris['sqrtA'].pow(2)
    n_0 = np.sqrt(mu / A.pow(3))
    n = n_0 + ephemeris['deltaN']
    M_k = ephemeris['M_0'] + n * sv_position['t_k']
    E_k = M_k
    err = pd.Series(data=[1] * len(sv_position.index))
    i = 0
    while err.abs().min() > 1e-8 and i < 10:
        new_vals = M_k + ephemeris['e'] * np.sin(E_k)
        err = new_vals - E_k
        E_k = new_vals
        i += 1

    sinE_k = np.sin(E_k)
    cosE_k = np.cos(E_k)
    delT_r = F * ephemeris['e'].pow(ephemeris['sqrtA']) * sinE_k
    delT_oc = transmit_time - ephemeris['t_oc']
    sv_position['delT_sv'] = ephemeris['SVclockBias'] + ephemeris['SVclockDrift'] * delT_oc + ephemeris[
        'SVclockDriftRate'] * delT_oc.pow(2)

    v_k = np.arctan2(np.sqrt(1 - ephemeris['e'].pow(2)) * sinE_k, (cosE_k - ephemeris['e']))

    Phi_k = v_k + ephemeris['omega']

    sin2Phi_k = np.sin(2 * Phi_k)
    cos2Phi_k = np.cos(2 * Phi_k)

    du_k = ephemeris['C_us'] * sin2Phi_k + ephemeris['C_uc'] * cos2Phi_k
    dr_k = ephemeris['C_rs'] * sin2Phi_k + ephemeris['C_rc'] * cos2Phi_k
    di_k = ephemeris['C_is'] * sin2Phi_k + ephemeris['C_ic'] * cos2Phi_k

    u_k = Phi_k + du_k

    r_k = A * (1 - ephemeris['e'] * np.cos(E_k)) + dr_k

    i_k = ephemeris['i_0'] + di_k + ephemeris['IDOT'] * sv_position['t_k']

    x_k_prime = r_k * np.cos(u_k)
    y_k_prime = r_k * np.sin(u_k)

    Omega_k = ephemeris['Omega_0'] + (ephemeris['OmegaDot'] - OmegaDot_e) * sv_position['t_k'] - OmegaDot_e * ephemeris[
        't_oe']

    sv_position['x_k'] = x_k_prime * np.cos(Omega_k) - y_k_prime * np.cos(i_k) * np.sin(Omega_k)
    sv_position['y_k'] = x_k_prime * np.sin(Omega_k) + y_k_prime * np.cos(i_k) * np.cos(Omega_k)
    sv_position['z_k'] = y_k_prime * np.sin(i_k)
    return sv_position


def least_squares(xs, measured_pseudorange, x0, b0):
    dx = 100*np.ones(3)
    b = b0
    # set up the G matrix with the right dimensions. We will later replace the first 3 columns
    # note that b here is the clock bias in meters equivalent, so the actual clock bias is b/LIGHTSPEED
    G = np.ones((measured_pseudorange.size, 4))
    iterations = 0
    while np.linalg.norm(dx) > 1e-3:
        # Eq. (2):
        r = np.linalg.norm(xs - x0, axis=1)
        # Eq. (1):
        phat = r + b0
        # Eq. (3):
        deltaP = measured_pseudorange - phat
        G[:, 0:3] = -(xs - x0) / r[:, None]
        # Eq. (4):
        sol = np.linalg.inv(np.transpose(G) @ G) @ np.transpose(G) @ deltaP
        # Eq. (5):
        dx = sol[0:3]
        db = sol[3]
        x0 = x0 + dx
        b0 = b0 + db
    norm_dp = np.linalg.norm(deltaP)
    return x0, b0, norm_dp


with open(input_filepath) as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
        if row[0][0] == '#':
            if 'Fix' in row[0]:
                android_fixes = [row[1:]]
            elif 'Raw' in row[0]:
                measurements = [row[1:]]
        else:
            if row[0] == 'Fix':
                android_fixes.append(row[1:])
            elif row[0] == 'Raw':
                measurements.append(row[1:])

android_fixes = pd.DataFrame(android_fixes[1:], columns = android_fixes[0])
measurements = pd.DataFrame(measurements[1:], columns = measurements[0])


# Format satellite IDs
measurements.loc[measurements['Svid'].str.len() == 1, 'Svid'] = '0' + measurements['Svid']
measurements.loc[measurements['ConstellationType'] == '1', 'Constellation'] = 'G'
measurements.loc[measurements['ConstellationType'] == '3', 'Constellation'] = 'R'
measurements['SvName'] = measurements['Constellation'] + measurements['Svid']

# Remove all non-GPS measurements
measurements = measurements.loc[measurements['Constellation'] == 'G']

# Convert columns to numeric representation
measurements['Cn0DbHz'] = pd.to_numeric(measurements['Cn0DbHz'])
measurements['TimeNanos'] = pd.to_numeric(measurements['TimeNanos'])
measurements['FullBiasNanos'] = pd.to_numeric(measurements['FullBiasNanos'])
measurements['ReceivedSvTimeNanos']  = pd.to_numeric(measurements['ReceivedSvTimeNanos'])
measurements['PseudorangeRateMetersPerSecond'] = pd.to_numeric(measurements['PseudorangeRateMetersPerSecond'])
measurements['ReceivedSvTimeUncertaintyNanos'] = pd.to_numeric(measurements['ReceivedSvTimeUncertaintyNanos'])

# A few measurement values are not provided by all phones
# We'll check for them and initialize them with zeros if missing
if 'BiasNanos' in measurements.columns:
    measurements['BiasNanos'] = pd.to_numeric(measurements['BiasNanos'])
else:
    measurements['BiasNanos'] = 0
if 'TimeOffsetNanos' in measurements.columns:
    measurements['TimeOffsetNanos'] = pd.to_numeric(measurements['TimeOffsetNanos'])
else:
    measurements['TimeOffsetNanos'] = 0

print(measurements.columns)

measurements['GpsTimeNanos'] = measurements['TimeNanos'] - (measurements['FullBiasNanos'] - measurements['BiasNanos'])
gpsepoch = datetime(1980, 1, 6, 0, 0, 0)
measurements['UnixTime'] = pd.to_datetime(measurements['GpsTimeNanos'], utc = True, origin=gpsepoch)
measurements['UnixTime'] = measurements['UnixTime']

# Split data into measurement epochs
measurements['Epoch'] = 0
measurements.loc[measurements['UnixTime'] - measurements['UnixTime'].shift() > timedelta(milliseconds=200), 'Epoch'] = 1
measurements['Epoch'] = measurements['Epoch'].cumsum()

WEEKSEC = 604800
LIGHTSPEED = 2.99792458e8

# This should account for rollovers since it uses a week number specific to each measurement

measurements['tRxGnssNanos'] = measurements['TimeNanos'] + measurements['TimeOffsetNanos'] - (measurements['FullBiasNanos'].iloc[0] + measurements['BiasNanos'].iloc[0])
measurements['GpsWeekNumber'] = np.floor(1e-9 * measurements['tRxGnssNanos'] / WEEKSEC)
measurements['tRxSeconds'] = 1e-9*measurements['tRxGnssNanos'] - WEEKSEC * measurements['GpsWeekNumber']
measurements['tTxSeconds'] = 1e-9*(measurements['ReceivedSvTimeNanos'] + measurements['TimeOffsetNanos'])
# Calculate pseudorange in seconds
measurements['prSeconds'] = measurements['tRxSeconds'] - measurements['tTxSeconds']

# Conver to meters
measurements['PrM'] = LIGHTSPEED * measurements['prSeconds']
measurements['PrSigmaM'] = LIGHTSPEED * 1e-9 * measurements['ReceivedSvTimeUncertaintyNanos']

manager = EphemerisManager(ephemeris_data_directory)

epoch = 0
num_sats = 0
while num_sats < 5 :
    one_epoch = measurements.loc[(measurements['Epoch'] == epoch) & (measurements['prSeconds'] < 0.1)].drop_duplicates(subset='SvName')
    timestamp = one_epoch.iloc[0]['UnixTime'].to_pydatetime(warn=False)
    one_epoch.set_index('SvName', inplace=True)
    num_sats = len(one_epoch.index)
    epoch += 1

sats = one_epoch.index.unique().tolist()
ephemeris = manager.get_ephemeris(timestamp, sats)
print(timestamp)
print(one_epoch[['UnixTime', 'tTxSeconds', 'GpsWeekNumber']])

sv_position = calculate_satellite_position(ephemeris, one_epoch['tTxSeconds'])
print(sv_position)

#initial guesses of receiver clock bias and position
b0 = 0
x0 = np.array([0, 0, 0])
xs = sv_position[['x_k', 'y_k', 'z_k']].to_numpy()

# Apply satellite clock bias to correct the measured pseudorange values
pr = one_epoch['PrM'] + LIGHTSPEED * sv_position['delT_sv']
pr = pr.to_numpy()

st_file = pd.DataFrame()
#pd.options.display.float_format = '{:.8f}'.format
st_file['GPS-time'] = one_epoch['UnixTime']
st_file['Sat.X'] = sv_position['x_k']
st_file['Sat.Y'] = sv_position['y_k']
st_file['Sat.Z'] = sv_position['z_k']
st_file['Pseudo-Range'] = pr
st_file['CN0'] = one_epoch['Cn0DbHz']

cur_dir = os.getcwd()
csv_file_name = "st_file.csv"
csv_file_path = os.path.join(cur_dir, csv_file_name)
st_file.to_csv(csv_file_path, index=True)




x, b, dp = least_squares(xs, pr, x0, b0)
print(navpy.ecef2lla(x))
print(b/LIGHTSPEED)
print(dp)

ecef_list = []
for epoch in measurements['Epoch'].unique():
    one_epoch = measurements.loc[(measurements['Epoch'] == epoch) & (measurements['prSeconds'] < 0.1)]
    one_epoch = one_epoch.drop_duplicates(subset='SvName').set_index('SvName')
    if len(one_epoch.index) > 4:
        timestamp = one_epoch.iloc[0]['UnixTime'].to_pydatetime(warn=False)
        sats = one_epoch.index.unique().tolist()
        ephemeris = manager.get_ephemeris(timestamp, sats)
        sv_position = calculate_satellite_position(ephemeris, one_epoch['tTxSeconds'])

        xs = sv_position[['x_k', 'y_k', 'z_k']].to_numpy()
        pr = one_epoch['PrM'] + LIGHTSPEED * sv_position['delT_sv']
        pr = pr.to_numpy()

        x, b, dp = least_squares(xs, pr, x, b)
        ecef_list.append(x)

# Perform coordinate transformations using the Navpy library

ecef_array = np.stack(ecef_list, axis=0)
lla_array = np.stack(navpy.ecef2lla(ecef_array), axis=1)

# Extract the first position as a reference for the NED transformation
ref_lla = lla_array[0, :]
ned_array = navpy.ecef2ned(ecef_array, ref_lla[0], ref_lla[1], ref_lla[2])

# Convert back to Pandas and save to csv
lla_df = pd.DataFrame(lla_array, columns=['Latitude', 'Longitude', 'Altitude'])
ned_df = pd.DataFrame(ned_array, columns=['N', 'E', 'D'])

ecef_df = pd.DataFrame(ecef_array, columns=['Pos.X', 'Pos.Y', 'Pos.Z'])
ecef_df = ecef_df / 1000
combined_df = pd.concat([lla_df, ecef_df], axis=1)
combined_df.to_csv(csv_file_name, index=True,mode='a')
print(combined_df)

kml = simplekml.Kml()
for index, row in lla_df.iterrows():
    kml.newpoint(name=f"Point {index}", coords=[(row['Longitude'], row['Latitude'], row['Altitude'])])
kml.save("lla_points.kml")

