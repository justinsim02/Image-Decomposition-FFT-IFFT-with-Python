import numpy as np
import matplotlib.pyplot as plt

# CREATING A SINUSOIDAL GRATING
#------------------------------------------
# x = np.arange(-500, 501, 1)
# X, Y = np.meshgrid(x, x)
# wavelength = 20
# grating = np.sin(2*np.pi * X /wavelength)

# plt.set_cmap("gray")
# plt.imshow(grating)
# plt.show()
#------------------------------------------
# TRANSFORMING THE AXES / ROTATION
#------------------------------------------
# x = np.arange(-500, 501, 1)
# X, Y = np.meshgrid(x, x)
# wavelength = 20
# angle = np.pi / 9
# grating = np.sin ( 2*np.pi*(X*np.cos(angle) + 
#                 Y*np.sin(angle)) / wavelength)
# plt.set_cmap("gray")
# plt.imshow(grating)
# plt.show()
#-----------------------------------------
# FOURIER TRANSFORMS / SUM GRATINGS
#-----------------------------------------
# x = np.arange(-500, 501, 1)
# X, Y = np.meshgrid(x, x)

# amplitudes = 0.5,0.25,1,0.75,1
# wavelengths = 200,100,250,300,60
# angles = 0, np.pi/4, np.pi/9, np.pi/2, np.pi/12

# #Sum of Gratings
# gratings = np.zeros(X.shape)
# for amp, w_len, angle in zip(amplitudes, wavelengths, angles):
#     gratings += amp * np.sin(
#         2*np.pi*(X*np.cos(angle) + Y*np.sin(angle)) / w_len)

# # Calculate Fourier Transform of sum of the gratings
# ft = np.fft.ifftshift(gratings)
# ft = np.fft.fft2(ft)
# ft = np.fft.fftshift(ft)

# plt.set_cmap("gray")
# plt.subplot(121)
# plt.imshow(gratings)
# # Calculate Fourier transform of grating

# plt.subplot(122)
# plt.imshow(abs(ft))
# plt.xlim([480, 520])
# plt.ylim([520, 480])  # Note, order is reversed for y
# plt.show()
#-----------------------------------------
# INVERSE FOURIER TRANSFORM
#-----------------------------------------
x = np.arange(-500,500,1)
X, Y = np.meshgrid(x,x)
wavelength = 100
angle = np.pi/9
grating = np.sin(
    2*np.pi*(X*np.cos(angle) + Y*np.sin(angle)) / wavelength
)

plt.set_cmap("gray")

plt.subplot(131)
plt.imshow(grating)
plt.axis("off")

# Calculate the Fourier Transform of a grating
ft = np.fft.ifftshift(grating)
ft = np.fft.fft2(ft)
ft = np.fft.fftshift(ft)

plt.subplot(132)
plt.imshow(abs(ft))
plt.axis("off")
plt.xlim([480,520])
plt.ylim([520,480])

#Calculate the Inverse Fourier Transform
ift = np.fft.ifftshift(ft)
ift = np.fft.ifft2(ift)
ift = np.fft.fftshift(ift)
ift = ift.real

plt.subplot(133)
plt.imshow(ift)
plt.axis("off")
plt.show()