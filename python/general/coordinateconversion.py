import numpy as np

class CoordinateConversion:
    
    def radial_distance(self, x, y, z):
        return np.linalg.norm([x, y, z])

class CartesianToSpherical(CoordinateConversion):
    def convert(self, x, y, z):
        r = self.radial_distance(x, y, z)
        x = np.float64(x)
        y = np.float64(y)
        z = np.float64(z)

        phi = np.arccos(z/r) #Inclination
        theta = np.arctan2(y,x) #Azimuthal
        return r, theta, phi

if __name__=='__main__':
    CoordinateConversion().radial_distance(0, 0, 1)
