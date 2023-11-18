import numpy as np
import matplotlib.pyplot as plt
import util



class Hyperbolic:
    def __init__(self, a: float, eccentricity: float, inclination: float, long_asc_node: float, arg_periapsis: float, mu: float):

        self.a = a
        self.eccentricity = eccentricity
        self.inclination = np.radians(inclination)
        self.long_asc_node = np.radians(long_asc_node)
        self.arg_periapsis = np.radians(arg_periapsis)
        self.mu = mu


    def basicKeplerEquationSolver( mean_anomaly: float, eccentricity: float) -> float:
            relative_change_epsilon: float=1e-6
            max_iter = 100
            # Initial starting guess for the hyperbolic anomaly.
            hyper_anomaly = 0.0
            if eccentricity >1:
                hyper_anomaly = np.pi

            # Iterate using Newton-Raphson.
            iter = 0
            relative_change = 1e6
            while (relative_change>relative_change_epsilon) and (iter< max_iter):

                delta_hyper_anomaly = (mean_anomaly - eccentricity*np.sinh(hyper_anomaly) + hyper_anomaly)/(1-eccentricity*np.cosh(hyper_anomaly))
                hyper_anomaly -= delta_hyper_anomaly
                relative_change = np.abs(delta_hyper_anomaly/(hyper_anomaly+1e-300))
                iter += 1
                # print(iter, hyper_anomaly, delta_hyper_anomaly)

            return hyper_anomaly
    

    def calculate_state(self):

        self.hyperbolic_anomaly = self.kepler_solver(self.mean_anomaly, self.eccentricity)
        self.true_anomaly = 2.0*np.arctan(np.sqrt((self.eccentricity+1)/(self.eccentricity-1))*np.tan(self.hyperbolic_anomaly*0.5))
        self.flight_path_angle = np.arctan2((self.eccentricity*np.sin(self.true_anomaly))/ (1.0+self.eccentricity*np.cos(self.true_anomaly)))

        r = self.a*(1-self.eccentricity*np.cosh(self.hyperbolic_anomaly))
        v = util.vis_viva(self.mu, r, self.a)

        pos_angle = self.true_anomaly + self.arg_periapsis
        vel_angle = pos_angle - self.flight_path_angle

        s = [r*(np.cos(pos_angle)*np.cos(self.long_asc_node) - np.sin(pos_angle)*np.cos(self.inclination)*np.sin(self.long_asc_node)),
            r*(np.cos(pos_angle)*np.sin(self.long_asc_node) + np.sin(pos_angle)*np.cos(self.inclination)*np.cos(self.long_asc_node))]

        return s