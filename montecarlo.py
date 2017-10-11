import numpy as np
import matplotlib.pyplot as plt

theta_min = 0.0
theta_max = np.pi / 2
phi_min = 0.0
phi_max = 2.0 * np.pi
spec_pow = 10
surface_normal = np.array([0, 0, 1])
in_samples = 8000
refl_samples = 1000


def blinn_phong(light_dir_in, light_dir_out):
    half_vector = light_dir_in + light_dir_out
    half_vector = half_vector / np.linalg.norm(half_vector) # Normalize
    n_dot_h = np.dot(surface_normal, half_vector)
    return n_dot_h ** spec_pow


def lambert(light_in_dir, light_out_dir):
    return 1


print("Generating Samples...")

incoming_zenith_angles = (theta_max - theta_min) * np.random.random_sample(in_samples) + theta_min
incoming_azimuth_angles = (phi_max - phi_min) * np.random.random_sample(in_samples) + phi_min
outgoing_zenith_angles = (theta_max - theta_min) * np.random.random_sample((in_samples, refl_samples)) + theta_min
outgoing_azimuth_angles = (phi_max - phi_min) * np.random.random_sample((in_samples, refl_samples)) + phi_min
integral_estimates = np.zeros(in_samples)

print("Performing Monte Carlo Integration...")

for i in range(in_samples):
    # Calculate incoming light vector (normalized 3 dimensional)
    theta_i = incoming_zenith_angles[i]
    phi_i = incoming_azimuth_angles[i]
    light_dir_in = np.array([
        0,
        0,
        1
    ])

    brdf_sum = 0
    j = 0
    for j in range(refl_samples):
        # Calculate outgoing light vector (normalized 3 dimensional)
        theta_r = outgoing_zenith_angles[i, j]
        phi_r = outgoing_azimuth_angles[i, j]
        light_dir_out = np.array([
            np.sin(theta_r) * np.cos(phi_r),
            np.sin(theta_r) * np.sin(phi_r),
            np.cos(theta_r)
        ])

        brdf_sum += blinn_phong(light_dir_in, light_dir_out) * np.cos(theta_r) * np.sin(theta_r)

    # Calculate integral estimate for given incoming light direction
    avg_brdf_value = brdf_sum / (j + 1)
    integral_estimates[i] = avg_brdf_value * (theta_max - theta_min) * (phi_max - phi_min)

    if (i % 100 == 0):
        print("{} samples processed".format(i + 1))

# Plot incoming zenith angle vs ratio of outgoing to incoming irradiance
plt.figure(figsize=(10,10))
plt.subplot(211)
ymean = [np.mean(integral_estimates)] * in_samples
max_idx = np.argmax(integral_estimates)
xmax = incoming_zenith_angles[max_idx]
ymax = integral_estimates[max_idx]
plt.plot(incoming_zenith_angles, integral_estimates, 'ro')
plt.plot(incoming_zenith_angles, ymean, '-')
plt.xlabel("Incoming Zenith Angle")
plt.ylabel("Ratio of outgoing irradiance to incoming irradiance")
plt.annotate('max: [{}, {}]'.format(xmax, ymax), xy=(xmax, ymax), xytext=(xmax + 0.1, ymax + 0.1),
             arrowprops=dict(facecolor='black', shrink=0.05))
plt.annotate('mean: {}'.format(ymean[0]), xy=(xmax, ymean[0]), xytext=(xmax + 0.1, ymean[0] + 0.1),
             arrowprops=dict(facecolor='black', shrink=0.05))

# Plot incoming azimuth angle vs ratio of outgoing to incoming irradiance
plt.subplot(212)
xmax = incoming_azimuth_angles[max_idx]
plt.plot(incoming_azimuth_angles, integral_estimates, 'ro')
plt.plot(incoming_azimuth_angles, ymean, '-')
plt.xlabel("Incoming Azimuth Angle")
plt.ylabel("Ratio of outgoing irradiance to incoming irradiance")
plt.annotate('max: [{}, {}]'.format(xmax, ymax), xy=(xmax, ymax), xytext=(xmax + 0.1, ymax + 0.1),
             arrowprops=dict(facecolor='black', shrink=0.05))
plt.annotate('mean: {}'.format(ymean[0]), xy=(xmax, ymean[0]), xytext=(xmax + 0.1, ymean[0] + 0.1),
             arrowprops=dict(facecolor='black', shrink=0.05))

plt.show()