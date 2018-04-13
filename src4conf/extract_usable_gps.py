import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
from pyproj import Geod

# from sklearn.preprocessing import StandardScaler
# from mpl_toolkits.mplot3d import Axes3D
import h5py

def data_norm_traj(gps_points):
    """
    @gps_points : list of gps points
    reference URL:
    https://qiita.com/tomo001/items/84d49e18e9dd3373ce0f
    """
    # paramters
    output = []  # output -> meter from the center
    dist_list = []
    g = Geod(ellps='WGS84')

    # find the central (basis)
    N = len(gps_points)/2
    c_lat, c_long, c_alt = gps_points[N]

    # (now - N*T) < time < now
    past_points = gps_points[:N]
    for n_lat, n_long, n_alt in past_points:

        # if the data is same as the center, simply consider it as origin
        if abs(c_long - n_long) < 0.00001 and abs(c_lat - n_lat) < 0.00001:
            output.append([0, 0, 0])
        else:
            result = g.inv(c_long, c_lat, n_long, n_lat)
            azimuth = result[0] * math.pi / 180.0
            distance_2d = result[2]
            x = distance_2d * math.cos(math.pi/2. - azimuth)
            y = distance_2d * math.sin(math.pi/2. - azimuth)
            z = n_alt - c_alt
            output.append([x, y, z])

    # time = central
    output.append([0., 0., 0.])

    # now < time < (now + N*T)
    future_points = gps_points[N+1:]
    for n_lat, n_long, n_alt in future_points:
        if abs(c_long - n_long) < 0.00001 and abs(c_lat - n_lat) < 0.00001:
            output.append([0, 0, 0])
        else:
            result = g.inv(c_long, c_lat, n_long, n_lat)
            azimuth = result[0] * math.pi / 180.0
            distance_2d = result[2]
            x = distance_2d * math.cos(math.pi/2. - azimuth)
            y = distance_2d * math.sin(math.pi/2. - azimuth)
            z = n_alt - c_alt
            output.append([x, y, z])

    # coordinate rotation
    top_point = output[-1]
    theta = math.atan2(top_point[1], top_point[0]) - math.pi / 2.0
    output_norm = []
    for out in output:
        new_out_x = out[0] * math.cos(theta) + out[1] * math.sin(theta)
        new_out_y = out[0] * -math.sin(theta) + out[1] * math.cos(theta)
        output_norm.append([new_out_x, new_out_y, out[2]])

    # make the distance list
    for i in range(int(2*N)):
        p_prev = np.array(output[i])
        p_next = np.array(output[i+1])
        dist_list.append(np.sqrt(np.power(p_prev - p_next, 2).sum()))

    return (output, dist_list, output_norm)


def plot_coordinate(coordinates, new_coordinates):
    # original trajectory
    X = np.array(coordinates)[:, 0]
    Y = np.array(coordinates)[:, 1]
    plt.plot(X, Y)
    plt.plot(X, Y, marker='o')
    plt.xlim(-100, 100)
    plt.ylim(-100, 100)
    plt.grid()

    # after rotation
    for i, (x, y) in enumerate(zip(X, Y)):
        plt.annotate(str(i), (x, y))

    X_new = np.array(new_coordinates)[:, 0]
    Y_new = np.array(new_coordinates)[:, 1]
    plt.plot(X_new, Y_new)
    plt.plot(X_new, Y_new, marker='^', )
    plt.xlim(-100, 100)
    plt.ylim(-100, 100)

    for i, (x, y) in enumerate(zip(X_new, Y_new)):
        plt.annotate(str(i), (x, y))
    plt.show()


def plot_3d(coordinates, filename):
    # original trajectory
    X = np.array(coordinates)[:, 0]
    Y = np.array(coordinates)[:, 1]
    Z = np.array(coordinates)[:, 2]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(X, Y, Z, "-", color="g", lw=5)
    ax.set_title(filename)
    plt.xlim(-100, 100)
    plt.ylim(-100, 100)
    ax.set_zlim(-10, 10)
    plt.grid()
    plt.savefig(filename, dpi=300)


# set parameters
N = 5

# read GPS data
GPS_data = pd.read_csv('./usable_data.csv')
GPS_data = GPS_data.ix[:, ['Latitude', 'Longitude', 'Altitude']]
count_false = 0  # okay data
count_true = 0  # non-okay data

out_gps = []

for i in range(GPS_data.shape[0] - N):

    # End Condition
    if i == 68840:
        break

    # debug
    print(i)
    coordinates, dist_list, new_coordinates = data_norm_traj(np.array(GPS_data[i:i+2*N+1]))

    # ignore the data which has more than 120km/h = 33m/sec (gps noise jump)
    # GPS does not observe data every 1 sec, maximum we need to consider 2 sec
    if max(dist_list) > 66.0:
        count_false += 1
        continue
    else:
        count_true += 1

    # add the non-noisy data to outcome array
    out_gps.append(new_coordinates)
    print(new_coordinates)

#    if i==100:
#        break

    # # plot samples
    # if i == 49600:
    #     plot_3d(new_coordinates, 'straight')
    # if i == 42100:
    #     plot_3d(new_coordinates, 'uphill')
    # if i == 63500:
    #     plot_3d(new_coordinates, 'downhill')
    # if i == 28600:
    #     plot_3d(new_coordinates, 'uphill_curve')
    # if i == 32194:
    #     plot_3d(new_coordinates, 'downhill_curve')

# save data
h5f = h5py.File('gps_standardized_data.h5', 'w')
h5f.create_dataset('data', data=out_gps)
h5f.close()
