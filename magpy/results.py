from scipy.integrate import trapz
import matplotlib.pyplot as plt


class Results:
    """Results of a simulation of a single particle cluster

    The results contain the time-varying magnetisation and field resulting from
    stochastic simulation of a particle cluster consisting of `N` particles.

    Args:
        time (np.ndarray): 1d array of length `M`. Time in seconds for
            each sample in the results
        field (np.ndarray): 1d array of length `M`. Field amplitude at each point
            in time. Field is always applied along the z-axis.
        x (dict): `{0: np.ndarray, ..., N-1: np.ndarray}` key, value pair is an interger
            particle id and a corresponding 1d array of length `M` for each of the
            `N` particles in the cluster. 1d array is the x-coordinate of the particle
            magnetisation vector at each point in time.
        y (dict): `{0: np.ndarray, ..., N-1: np.ndarray}` key, value pair is an interger
            particle id and a corresponding 1d array of length `M` for each of the
            `N` particles in the cluster. 1d array is the y-coordinate of the particle
            magnetisation vector at each point in time.
        z (dict): `{0: np.ndarray, ..., N-1: np.ndarray}` key, value pair is an interger
            particle id and a corresponding 1d array of length `M` for each of the
            `N` particles in the cluster. 1d array is the z-coordinate of the particle
            magnetisation vector at each point in time.
        N (int): number of particles in the ensemble
    """
    def __init__(self, time, field, x, y, z, N):
        self.time = time
        self.field = field
        self.x = x
        self.y = y
        self.z = z
        self.N = N

    def plot(self):
        """Plots the magnetisation from the results

        Plots the x,y,z coordinates of the magnetisation vector for every particle
        in the particle cluster.

        Returns:
            matplotlib figure handle containing the resulting plot axes.
        """
        fg, axs = plt.subplots(nrows=self.N)
        if self.N==1:
            axs = [axs]
        for idx in range(self.N):
            axs[idx].plot(self.time, self.x[idx], label='x')
            axs[idx].plot(self.time, self.y[idx], label='y')
            axs[idx].plot(self.time, self.z[idx], label='z')
            axs[idx].legend()
            axs[idx].set_title('Particle {}'.format(idx))
            axs[idx].set_xlabel('Reduced time [dimless]')
            fg.tight_layout()
        return fg

    def magnetisation(self, direction='z'):
        """Computes the total magnetisation of the cluster

        Computes the total time-varying magnetisation of the cluster in a desired
        direction. The total magnetisation is simply the sum of the individual
        magnetisation vector components in the specified direction (x,y, or z).

        Args:
            direction (str, optional): the direction of magnetisation `x`, `y` or `z`.
                Default value is `z`.
        Returns:
            np.ndarray: 1d array of length `M` the total magnetisation at each point
                in `self.time`.
        """
        return np.sum([vals for vals in getattr(self, direction).values()], axis=0)

    def final_state(self):
        """The state of the cluster at the end of the simulation.

        Returns the state of the particle cluster at the end of the simulation time.

        Returns:
            dict: a nested dictionary `{'x': {0: m_x, ..., N-1: m_x}, 'y': ...}`
                containing the final value of the magnetisation vector for each
                of the `N` particles in the cluster.
        """
        return {
            'x': {k:v[-1] for k,v in self.x.items()},
            'y': {k:v[-1] for k,v in self.y.items()},
            'z': {k:v[-1] for k,v in self.z.items()}
        }


class EnsembleResults:
    """Results from a simulation of an ensemble of particle clusters

    The EnsembleResults object holds the resulting `magpy.Results` objects for
    an ensemble of simulated particle clusters. It provides a user-friendly
    alternative to handling a large collection of `magpy.Results` instances and
    implemetns methods for computing ensemble-wide properties.

    Args:
        results (list[magpy.Results]): results for each particle cluster in the ensemble

    Attributes:
        time: (np.ndarray): 1d array of length `M`. Time in seconds for each sample
            in the ensemble results.
        field: (np.ndarray): 1d array of length `M`. Field amplitude at each point in time.
            Field is always applied along the z-axis.
    """
    def __init__(self, results):
        self.results = results
        self.time = results[0].time
        self.field = results[0].field

    def magnetisation(self, direction='z'):
        """Total magnetisation of each member of the ensemble

        The total magnetisation of cluster is computed by summing the components of
        the magnetisation vector for each particle in the cluster. The component (`x`,`y`,`z`)
        along which the magnetisation may be specified. The default value is `z`,
        which is the same direction as the applied magnetic field.

        Args:
            direction (str, optional): direction of magnetisation `x`, `y` or `z`.
                Default value is `z`.

        Returns:
            list[np.ndarray]: list containing a length `M` 1d array containing
            the total magnetisation of each particle cluster in the ensemble.
        """
        return [res.magnetisation(direction) for res in self.results]

    def ensemble_magnetisation(self, direction='z'):
        """Total magnetisation of entire ensemble

        The total magnetisation of an ensemble of particle clusters. The ensemble
        magnetisation is the average value of the magnetisation of each particle
        particle cluster in the ensemble at each point in time. The component (`x`,`y`,`z`)
        along which the magnetisation may be specified. The default value is `z`,
        which is the same direction as the applied magnetic field.

        Args:
            direction (str, optional): direction of magnetisation `x`, `y` or `z`.
                Default value is `z`.

        Returns:
            np.ndarray: 1d array of length `M` containing the ensemble magnetisation
                for each point in `self.time`
        """
        return np.sum(self.magnetisation(direction), axis=0) / len(self.results)

    def final_state(self):
        """State of each ensemble member at the end of the simulation.

        The final state of each particle cluster in the ensemble at the end
        of the simulation time. The state of each particle cluster is the value
        of magnetisation vector of every particle in the cluster.

        Returns:
            list[dict]: a list of nested dictionaries like `{'x': {0: m_x, ..., N-1: m_x}, 'y': ...}`.
            The dictionaries contain the final value of the magnetisation vector for
            each of the `N` particles in the cluster.
        """
        return [res.final_state() for res in self.results]

    def energy_dissipated(self, start_time=None, end_time=None):
        """Total energy dissipated by the ensemble.

        A simulation with a constant or zero applied field will
        dissipate no energy. The energy dissipated by an ensemble of
        magnetic particle clusters subjected to an alternating field
        is the area of the hysteresis loop (magnetisation-field
        plane).

        The energy dissipated may be computed for the entire simulation
        or within a specific time window, defined by `start_time` and `end_time`

        Args:
            start_time (double, optional): the start of the time window for computing energy dissipated.
                Default value `None` uses the start of the simulation.
            end_time (double, optional): the end of the time window for computing energy dissipated.
                Default value `None` uses the end of the simulation.

        Returns:
            double: total energy dissipated by the ensemble during the time window
        """
        before_mask = (self.time >= start_time) if start_time is not None else True
        after_mask = (self.time <= end_time) if end_time is not None else True
        mask = before_mask & after_mask
        return -get_mu0() * trapz(self.field[mask], self.ensemble_magnetisation()[mask])

    def final_cycle_energy_dissipated(self, field_frequency):
        """Energy dissipated by the final cycle of the magnetic field.

        A simulation with a constant or zero applied field will
        dissipate no energy. The energy dissipated by an ensemble of
        magnetic particle clusters subjected to an alternating field
        is the area of the hysteresis loop (magnetisation-field
        plane).

        Use this function to compute the energy dissipated by the final
        cycle (i.e. period) of the applied alternating magnetic field
        if the total simulation time contains multiple cycles of the field
        (i.e. is longer than the period of the applied field). A common
        use case for this is to simulate a large number field cycles to
        reach equilibrium and then compute the energy dissipated during a
        single cycle of the field in equilibrium.

        Args:
            field_frequency (double): the frequency of the applied magnetic field

        Returns:
            double: energy dissipated during the last cycle of the applied magnetic field.
        """
        T = 1./field_frequency
        return self.energy_dissipated(start_time=self.time[-1] - T)
