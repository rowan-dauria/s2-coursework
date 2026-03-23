import numpy as np
from scipy.stats import gamma, poisson
from scipy.special import gammaln, logsumexp
from tqdm import tqdm
from copy import deepcopy


class RJMCMC:
    r"""
    Class for performing Reversible Jump MCMC (RJMCMC) on the change point
    model for the rate of serious mining accidents.

    This follows the approach described in Green (1995).

    Examples
    --------
    >>> intervals = np.loadtxt('coal_mining_accident_data.dat',
                                dtype=int, comments='#').T.flatten()
    >>> model = RJMCMC(intervals)
    >>> model.run_mcmc(num_iter=100)

    The resulting Markov chain is stored in the list `model.chain`.

    >>> print(model.chain)

    .. math::
        \{ x_{0}, x_{1}, x_{2}, \ldots, x_{\rm num iter} \}

    Note that becaused the model dimension varies, the entries in `model.chain`
    are generically arrays with different shapes.

    The model parameters are arrays with the first `k` entries being the change
    point times [days], and the next `k+1` entries being the Poisson rates for
    each segment [accidents/day];

    ... math::
        x = (s_1, s_2, \ldots, s_k, h_0, h_1, \ldots, h_k).
    """

    def __init__(self,
                 intervals,
                 lam=3.0,
                 k_max=30,
                 alpha=1.0,
                 beta=200.0):
        r"""
        Parameters
        ----------
        intervals : ndarray
            Array of time intervals between events [days]. Must be non-negative.
            Array has shape is (N-1,) and has dtype=int.
        lam : float
            Prior on the model :math:`k\sim\mathrm{Poisson}(\lambda)`
            conditioned on :math:`k \leq k_{\max}`. Optional.
        k_max : int, optional
            Maximum number of possible change points. Optional.
        alpha : float, optional
            Prior on the rates :math:`h_j\sim\mathrm{Gamma}(\alpha, \beta)`.
            Shape parameter for the Gamma prior. Optional.
        beta : float, optional
            Prior on the rates :math:`h_j\sim\mathrm{Gamma}(\alpha, \beta)`.
            Rate parameter for the Gamma prior. Optional.
        """
        self.intervals = np.array(intervals, dtype=int)

        assert np.all(self.intervals >= 0), \
            "All time intervals must be non negative."

        # Number of accidents, N
        N_minus_1, = self.intervals.shape
        self.num_accidents = 1 + N_minus_1

        # Accident times, y_i
        self.accident_times = np.hstack([0, np.cumsum(self.intervals)])

        # Total duration, L
        self.duration = np.sum(self.intervals)

        # Prior parameters
        self.lam = float(lam)
        self.k_max = int(k_max)
        self.alpha = float(alpha)
        self.beta = float(beta)

        # Prior on the model index k
        self.prior_k = poisson(self.lam)

        # Prior on the Poisson rates h_j
        self.rate_prior = gamma(a=self.alpha,
                                scale=1.0/self.beta)

        # Compute constant c for RJMCMC proposal
        self.compute_c()

    def compute_c(self):
        r"""
        Compute constant `c` for the RJMCMC proposal.
        """
        k_vals = np.arange(0, self.k_max + 1)

        b_k = np.array([ min(1, self.prior_k.pmf(k+1)/self.prior_k.pmf(k))
                        if k<self.k_max else 0.0 for k in k_vals ])

        d_k = np.array([ min(1, self.prior_k.pmf(k-1)/self.prior_k.pmf(k))
                        if k>0 else 0.0 for k in k_vals ])

        self.c = 0.9/np.max(b_k + d_k)

    def log_likelihood(self,
                       x):
        r"""
        Computes the log-likelihood of the data given the model parameters `x`
        for the inhomogenous Poisson process with :math:`k` change points.

        .. math::
            \log \mathcal{L}(\{I_i\}|\{s_j\},\{h_j\},M_k) =
                \sum_{i=1}^{N} \log x(y_i) - \int_0^L \mathrm{d}t\; x(t)

        Parameters
        ----------
        x : ndarray
            Array of model parameters, shape=(self.num_dim,).

        Returns
        -------
        logL : float
            The log-likelihood of the data given the model parameters.
        """
        x = np.asarray(x)

        k = (len(x)-1) // 2

        change_points = x[0:k]      # s_1, s_2, ..., s_k
        poisson_rates = x[k:2*k+1]  # h_0, h_1, h_2, ..., h_k

        idx = np.searchsorted(change_points, self.accident_times, side="right")
        gaps = np.diff(np.hstack([0., change_points, self.duration]))

        logL = np.sum(np.log(poisson_rates[idx])) - np.sum(poisson_rates*gaps)

        return logL

    def transition(self,
                   state):
        r"""
        Metropolis-Hastings transition to the next state in the Markov chain.

        Parameters
        ----------
        state : ndarray
            Current state :math:`x_{i}` of the Markov chain.

        Returns
        -------
        next_state : ndarray
            Next state :math:`x_{i+1}` of the Markov chain.
        move_type : int
            Type of move performed: 0 = height change, 1 = position change,
            2 = birth, 3 = death. This keeps track of the move types attempted.
        accept : int
            This is used to track the acceptance fraction of the chain;
            1 if the move was accepted, 0 otherwise.
        """
        k = (len(state) - 1) // 2

        b_k = self.c * min(1, self.prior_k.pmf(k+1)/self.prior_k.pmf(k)) \
                if k<self.k_max else 0.0
        d_k = self.c * min(1, self.prior_k.pmf(k-1)/self.prior_k.pmf(k)) \
                if k>0 else 0.0

        if k==0:
            eta_k = 1.0 - b_k - d_k
            pi_k = 0.0
        else:
            eta_k = 0.5 * (1.0 - b_k - d_k)
            pi_k = 0.5 * (1.0 - b_k - d_k)

        assert np.isclose(eta_k + pi_k + b_k + d_k, 1.0), \
            "Move probabilities do not sum to 1."

        # POSSIBLE MOVE TYPES:
        # 0 : H = height change move, with probability eta_k
        # 1 : P = position change, with probability pi_k
        # 2 : k = birth move, with probability b_k
        # 3 : k-1 = death move, with probability d_k
        move_type = np.random.choice(4, p=np.array([eta_k, pi_k, b_k, d_k]))

        if move_type == 0:
            next_state, accept = self.height_change_move(state)
        elif move_type == 1:
            next_state, accept = self.position_change_move(state)
        elif move_type == 2:
            next_state, accept = self.birth_move(state)
        elif move_type == 3:
            next_state, accept = self.death_move(state)
        else:
            raise ValueError("Invalid move type.")

        return next_state, move_type, accept

    def height_change_move(self,
                           state):
        r"""
        Height change move: propose to change one of the Poisson rates. The move
        is either accepted or rejected using the Metropolis-Hastings criterion.

        Parameters
        ----------
        state : ndarray
            Current state :math:`x_{i}` of the Markov chain.

        Returns
        -------
        next_state : ndarray
            Next state :math:`x_{i+1}` of the Markov chain.
        accept : int
            This is used to track the acceptance fraction of the chain;
            1 if the move was accepted, 0 otherwise.
        """
        k = (len(state) - 1) // 2
        proposed_state = deepcopy(state)

        # choose which height to change
        # index into the rates part of state
        j = np.random.randint(0, k+1)

        # isolate the height to be perturbed
        h_j = state[k+j]
        # log_ratio = log(h_j_prime/h_j) ~ U(-.5, .5)
        # from Green 1995
        log_ratio = np.random.uniform(-.5, .5)
        # perturb the heigh
        h_j_prime = h_j * np.exp(log_ratio)
        # update the proposed state with perturbed height
        proposed_state[k+j] = h_j_prime

        # compute log likelihood ratio of the perturbed state to the original state
        log_likelihood_ratio = self.log_likelihood(proposed_state) - \
                                self.log_likelihood(state)

        # log prior ratio + log Jacobian
        # prior ratio: (h_j'/h_j)^(alpha-1) * exp(-beta*(h_j'-h_j))
        # Jacobian of h_j' = h_j*exp(U) w.r.t. U is h_j'/h_j
        # combined: (h_j'/h_j)^alpha * exp(-beta*(h_j'-h_j))
        log_accept_prob = log_likelihood_ratio + \
                            self.alpha * np.log(h_j_prime / h_j) + \
                            -self.beta * (h_j_prime - h_j)

        # accept or reject with probability exp(log_accept_prob)
        if np.log(np.random.uniform()) < log_accept_prob:
            return proposed_state, 1
        else:
            return state, 0


    def position_change_move(self,
                             state,
                             eps=1.0e-10):
        r"""
        Position change move: propose to change the position of an existing
        change point. The move is either accepted or rejected using the
        Metropolis-Hastings criterion.

        Parameters
        ----------
        state : ndarray
            Current state :math:`x_{i}` of the Markov chain.
        eps : float
            Small tolerance to avoid proposing change points with zero length
            gaps. Optional.

        Returns
        -------
        next_state : ndarray
            Next state :math:`x_{i+1}` of the Markov chain.
        accept : int
            This is used to track the acceptance fraction of the chain;
            1 if the move was accepted, 0 otherwise.
        """
        k = (len(state) - 1) // 2
        proposed_state = deepcopy(state)

        # choose which position to use
        j = np.random.randint(1, k+1)

        # propose new position
        s_jminus1 = state[j-2] if j>1 else 0.0
        s_j = state[j-1]
        s_jplus1 = state[j] if j<k else self.duration
        s_j_prime = np.random.uniform(low=s_jminus1+eps, high=s_jplus1-eps)
        proposed_state[j-1] = s_j_prime

        # log likelihood ratio
        log_like_ratio = self.log_likelihood(proposed_state) - \
                            self.log_likelihood(state)

        # log acceptance probability
        log_accept_prob = log_like_ratio + \
                            np.log(s_jplus1-s_j_prime) + \
                            np.log(s_j_prime - s_jminus1) - \
                            np.log(s_jplus1 - s_j) - \
                            np.log(s_j - s_jminus1)

        # either accept or reject the proposed move
        if np.log(np.random.uniform()) < log_accept_prob:
            accept = 1
            return proposed_state, accept
        else:
            accept = 0
            return state, accept

    def birth_move(self,
                   state,
                   eps=1.0e-10):
        r"""
        Birth move: propose to add a new change point. This increases the
        model dimension. The move is either accepted or rejected using the
        Metropolis-Hastings criterion.

        Parameters
        ----------
        state : ndarray
            Current state :math:`x_{i}` of the Markov chain.
        eps : float
            Small tolerance to avoid any zero rates for stability. Optional.

        Returns
        -------
        next_state : ndarray
            Next state :math:`x_{i+1}` of the Markov chain.
        accept : int
            This is used to track the acceptance fraction of the chain;
            1 if the move was accepted, 0 otherwise.
        """
        k = (len(state) - 1) // 2

        change_points = np.hstack((state[0:k], np.array([self.duration])))
        poisson_rates = state[k:2*k+1]

        # choose new position
        s_star = np.random.uniform(low=0.0, high=self.duration)
        idx = np.argmax(s_star<change_points)
        s_j = change_points[idx-1] if idx>0 else 0.0
        s_jplus1 = change_points[idx]
        proposed_change_points = np.sort(np.hstack((state[0:k], s_star)))

        # choose new rate
        u = np.random.uniform(low=eps, high=1.0-eps)
        h_j = poisson_rates[idx]
        proposed_poisson_rates = deepcopy(poisson_rates)
        h_j_prime = h_j * ((1-u)/u)**(-(s_jplus1-s_star)/(s_jplus1 - s_j))
        h_jplus1_prime = h_j_prime * ((1-u)/u)
        proposed_poisson_rates[idx] = h_j_prime
        proposed_poisson_rates = np.insert(proposed_poisson_rates,
                                           idx+1, h_jplus1_prime)

        # proposed state
        proposed_state = np.hstack([proposed_change_points,
                                    proposed_poisson_rates])

        # log likelihood ratio
        log_like_ratio = self.log_likelihood(proposed_state) - \
                            self.log_likelihood(state)

        def _log_prior_ratio_birth():
            """
            Calculate the log ratio of the prior probabilities of even orders stats for the birth move.

            Is a sum of:
            * Log ratio of the *number* of change points
                * log p(k+1) - log p(k)
            * Log ratio of the *position* of the change points
                * log(π(s_1,...,s*,...,s_k | M_{k+1}) / π(s_1,...,s_k | M_k))
            * Log ratio of the *rate* between the change points
            """
            # log ratio of model priors: log p(k+1) - log p(k)
            log_model_ratio = (np.log(self.prior_k.pmf(k + 1))
                               - np.log(self.prior_k.pmf(k)))

            # log ratio of change point position priors (even-numbered order
            # statistics / Dirichlet): accounts for the new gap structure

            log_position_ratio = (np.log(2 * (k + 1) * (2 * k + 3))  # (2k+3)!/(2k+1)! = (2k+3)(2k+2)
                                  - 2 * np.log(self.duration) # -2log(L)
                                  + np.log(s_star - s_j)
                                  + np.log(s_jplus1 - s_star)
                                  - np.log(s_jplus1 - s_j))

            # log ratio of height priors: one extra Gamma(alpha, beta) factor
            # for the new rate, plus the change from h_j to (h_j', h_{j+1}')
            # NB: This can be simplified for alpha=1. Gamma(1) = 1 and alpha-1 = 0
            log_height_ratio = (self.alpha * np.log(self.beta)
                                - gammaln(self.alpha)
                                + (self.alpha - 1) * (np.log(h_j_prime)
                                                      + np.log(h_jplus1_prime))
                                - (self.alpha - 1) * np.log(h_j)
                                - self.beta * (h_j_prime + h_jplus1_prime - h_j))

            return log_model_ratio + log_position_ratio + log_height_ratio

        # log prior ratio
        log_prior_ratio = _log_prior_ratio_birth()

        # log proposal ratio
        bk = self.c * min(1, self.prior_k.pmf(k+1)/self.prior_k.pmf(k))
        dkplus1 = self.c * min(1, self.prior_k.pmf(k)/self.prior_k.pmf(k+1))
        log_proposal_ratio = np.log(dkplus1*self.duration) - np.log(bk*(k+1))

        # log Jacobian
        log_jacobian = 2.0 * np.log(h_jplus1_prime + h_j_prime) - np.log(h_j)

        # log acceptance probability
        log_accept_prob = log_like_ratio + log_prior_ratio + \
                                log_proposal_ratio + log_jacobian

        # either accept or reject the proposed move
        if np.log(np.random.uniform()) < log_accept_prob:
            accept = 1
            return proposed_state, accept
        else:
            accept = 0
            return state, accept

    def death_move(self,
                   state):
        r"""
        Death move: propose to remove an existing change point. This decreases
        the model dimension. The move is either accepted or rejected using the
        Metropolis-Hastings criterion.

        Parameters
        ----------
        state : ndarray
            Current state :math:`x_{i}` of the Markov chain.

        Returns
        -------
        next_state : ndarray
            Next state :math:`x_{i+1}` of the Markov chain.
        accept : int
            This is used to track the acceptance fraction of the chain;
            1 if the move was accepted, 0 otherwise.
        """
        k = (len(state) - 1) // 2

        change_points = state[0:k]
        poisson_rates = state[k:2*k+1]

        # choose which change point to remove
        j = np.random.randint(1, k+1)
        s_jminus1 = change_points[j-2] if j>1 else 0.0
        s_j = change_points[j-1]
        s_jplus1 = change_points[j] if j<k else self.duration
        proposed_change_points = np.delete(change_points, j-1)

        # propose new rate
        logh_j = np.log(poisson_rates[j-1])
        logh_jplus1 = np.log(poisson_rates[j])
        logh_j_prime = ( (s_j-s_jminus1)*logh_j + (s_jplus1-s_j)*logh_jplus1 ) / \
                    (s_jplus1 - s_jminus1)
        proposed_poisson_rates = np.delete(poisson_rates, j)
        proposed_poisson_rates[j-1] = np.exp(logh_j_prime)

        # proposed state
        proposed_state = np.hstack([proposed_change_points,
                                    proposed_poisson_rates])

        # log likelihood ratio
        log_like_ratio = self.log_likelihood(proposed_state) - \
                            self.log_likelihood(state)

        # log prior ratio
        def _log_prior_ratio_death():
            """
            Calculate the log ratio of the prior probabilities of even orders stats for the death move.

            Is a sum of:
            * Log ratio of the *number* of change points
                * log p(k-1) - log p(k)
            * Log ratio of the *position* of the change points
                * log(π(s_1,...,x,...,s_k | M_{k-1}) / π(s_1,...,s_k | M_k))
            * Log ratio of the *rate* between the change points
            """
            # log ratio of model priors: log p(k-1) - log p(k)
            log_model_ratio = (np.log(self.prior_k.pmf(k - 1))
                               - np.log(self.prior_k.pmf(k)))

            # log ratio of change point position priors (even-numbered order
            # statistics / Dirichlet): accounts for the new gap structure

            log_position_ratio = (2 * np.log(self.duration)
                                  - np.log(2 * k * (2 * k + 1))
                                  + np.log(s_jplus1 - s_jminus1)
                                  - np.log(s_j - s_jminus1)
                                  - np.log(s_jplus1 - s_j))

            # log ratio of height priors: one extra Gamma(alpha, beta) factor
            # for the new rate, plus the change from h_j to (h_j', h_{j+1}')
            # NB: This can be simplified for alpha=1. Gamma(1) = 1 and alpha-1 = 0, but keeping general
            log_height_ratio = (gammaln(self.alpha)
                                - self.alpha * np.log(self.beta)
                                + (self.alpha - 1) * logh_j_prime
                                - (self.alpha - 1) * (logh_j
                                                        + logh_jplus1)
                                + self.beta * (np.exp(logh_j_prime) # h^prime
                                                - poisson_rates[j-1] # h_j
                                                - poisson_rates[j])) # h_j+1

            return log_model_ratio + log_position_ratio + log_height_ratio

        log_prior_ratio = _log_prior_ratio_death()

        # log proposal ratio
        bkminus1 = self.c * min(1, self.prior_k.pmf(k)/self.prior_k.pmf(k-1))
        dk = self.c * min(1, self.prior_k.pmf(k-1)/self.prior_k.pmf(k))
        log_proposal_ratio = np.log(bkminus1*k) - np.log(dk*self.duration)

        # log Jacobian
        log_jacobian = logh_j_prime - 2.0 * logsumexp([logh_j,logh_jplus1])

        # log acceptance probability
        log_accept_prob = log_like_ratio + log_prior_ratio + \
                                log_proposal_ratio + log_jacobian

        # either accept or reject the proposed move
        if np.log(np.random.uniform()) < log_accept_prob:
            accept = 1
            return proposed_state, accept
        else:
            accept = 0
            return state, accept

    def run_mcmc(self,
                 num_iter=100):
        r"""
        Run the RJMCMC algorithm for a specified number of iterations.

        Parameters
        ----------
        num_iter : int
            Number of MCMC iterations to perform.
        """
        # initialise the MCMC at these parameter values
        x0 = np.array([1.4e+04, 8.6e-03, 2.6e-03])

        self.chain = [x0]
        move_type_attempt_count = {0:0, 1:0, 2:0, 3:0}
        move_type_accept_count = {0:0, 1:0, 2:0, 3:0}

        # evolve chain for specified number of iterations
        for i in tqdm(range(num_iter)):
            next_state, move_type, accepted = self.transition(self.chain[-1])
            self.chain.append(next_state)
            move_type_attempt_count[move_type] += 1
            move_type_accept_count[move_type] += accepted

        # print a summary of the move types and acceptance fractions
        for i in range(4):
            f = move_type_attempt_count[i]/num_iter
            a = move_type_accept_count[i]/max(1, move_type_attempt_count[i])
            m = ['Height change', 'Position change', 'Birth', 'Death'][i]
            print(f"{m} moves attempted {100*f:.1f}% of the time",
                f"with an acceptance fraction of {a:.3f}")


def main():
    intervals = np.loadtxt('coal_mining_accident_data.dat').T.flatten()
    sampler = RJMCMC(intervals)
    sampler.run_mcmc(num_iter=1000)


if __name__ == "__main__":

    main()