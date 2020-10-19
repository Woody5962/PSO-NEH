from functools import partial
from Gatt_Figure import gatt
import numpy as np
import random
import math


def _obj_wrapper(func, args, kwargs, x):
    return func(x, *args, **kwargs)


def _is_feasible_wrapper(func, x):
    return np.all(func(x) >= 0)


def _cons_none_wrapper(x):
    return np.array([0])


def _cons_ieqcons_wrapper(ieqcons, args, kwargs, x):
    return np.array([y(x, *args, **kwargs) for y in ieqcons])


def _cons_f_ieqcons_wrapper(f_ieqcons, args, kwargs, x):
    return np.array(f_ieqcons(x, *args, **kwargs))


def ms_init(lb, ub):
    """
	perform an initiation of the solution of machines
	:param lb: the lower bound of solution
	:param ub: the upper bound of solution
	:return: a solution that considering the balance of the distribution of machines
	"""
    ms = lb.copy()  # a original solution

    def arrange(k):
        """
		perform an arrangement of machines
		:param k: the id of a col
		:return: a better arrangement of the k-th col
		"""
        lb_k = lb[0, k]
        ub_k = ub[0, k]
        ms_k = ms[:, k].copy()
        n_machine = int(ub_k - lb_k + 1)  # number of machines
        n = len(ms_k)
        arrangement = {}
        for i_m in range(n_machine):
            index = lb_k + i_m  # index of machines
            arrangement[index] = []  # list of jobs
        for i_j in range(n):
            x = ms_k[i_j]
            arrangement[x].append(i_j)
        # arranged list A-Z according to number of jobs
        a_list = sorted(arrangement.items(), key=lambda item: len(item[1]))
        while len(a_list[0][1]) <= n / n_machine - 1:
            len_max = len(a_list[-1][1])
            random_job_id = random.randint(0, len_max - 1)  # guarantee the randomness
            key_max = a_list[-1][0]
            key_min = a_list[0][0]
            random_job = arrangement[key_max].pop(random_job_id)
            arrangement[key_min].append(random_job)
            a_list = sorted(arrangement.items(), key=lambda item: len(item[1]))
        # reorder ms_k
        for machine in arrangement:
            for job in arrangement[machine]:
                ms[job, k] = machine

    m = np.shape(ms)[1]
    for i in range(m):
        arrange(i)

    return ms


def pso(func, PT, mop, ieqcons=[], f_ieqcons=None, kwargs={},
        swarmsize=11, mutation_shreshold=3, maxiter=20, minfunc=1e-5,
        debug=False, debug_neh=False, processes=1, particle_output=False):
    """
	Perform a particle swarm optimization (PSO)

	Parameters
	==========
	func : function
		The function to be minimized
	PT : matrix
		the processing time matrix
	mop : array
		the number of machines of each procedure

	Optional parameters
	========
	ieqcons : list
		A list of functions of length n such that ieqcons[j](x,*args) >= 0.0 in
		a successfully optimized problem (Default: [])
	f_ieqcons : function
		Returns a 1-D array in which each element must be greater or equal
		to 0.0 in a successfully optimized problem. If f_ieqcons is specified,
		ieqcons is ignored (Default: None)
	kwargs : dict
		Additional keyword arguments passed to objective and constraint
		functions (Default: empty dict)
	swarmsize : int
		The number of particles in the swarm (Default: 100)
	mutation_shreshold : int
		The number of generations that the solution was improved very slowly
	maxiter : int
		The maximum number of iterations for the swarm to search (Default: 100)
	minfunc : scalar
		The minimum change of swarm's best objective value before the search
		terminates (Default: 1e-8)
	debug : boolean
		If True, progress statements will be displayed every iteration
		(Default: False)
	debug_neh: boolean
	    if true, m_neh will be displayed
	    (Default: False)
	processes : int
		The number of processes to use to evaluate objective function and
		constraints (default: 1)
	particle_output : boolean
		Whether to include the best per-particle position and the objective
		values at those.

	Returns
	=======
	g : array
		The swarm's best known position (optimal design)
	f : scalar
		The objective value at ``g``
	p : array
		The best known position per particle
	pf: arrray
		The objective values at each position in p

	"""

    n = np.shape(PT)[0]  # amount of jobs
    lb = np.zeros((n, len(mop)), dtype=int)  # The lower bounds of the design variable(s)
    ub = np.zeros_like(lb, dtype=int)  # The upper bounds of the design variable(s)
    for i in range(n):
        ub[i, 0] = mop[0] - 1
    for i in range(1, len(mop)):
        for j in range(n):
            lb[j, i] = lb[j, i - 1] + mop[i - 1]
            ub[j, i] = ub[j, i - 1] + mop[i]
    assert hasattr(func, '__call__'), 'Invalid function handle'
    assert np.all(lb <= ub), 'Wrong initial solution'

    # Initialize objective function
    args = (PT, mop, debug_neh)
    obj = partial(_obj_wrapper, func, args, kwargs)

    # Check for constraint function(s)
    if f_ieqcons is None:
        if not len(ieqcons):
            if debug:
                print('No constraints given.')
            cons = _cons_none_wrapper
        else:
            if debug:
                print('Converting ieqcons to a single constraint function')
            cons = partial(_cons_ieqcons_wrapper, ieqcons, args, kwargs)
    else:
        if debug:
            print('Single constraint function given in f_ieqcons')
        cons = partial(_cons_f_ieqcons_wrapper, f_ieqcons, args, kwargs)
    is_feasible = partial(_is_feasible_wrapper, cons)

    # Initialize the multiprocessing module if necessary
    if processes > 1:
        import multiprocessing
        mp_pool = multiprocessing.Pool(processes)

    # Initialize the particle swarm
    S = swarmsize
    X = []  # particle positions
    v = []  # particle velocities
    p = []  # best particle positions
    for particle in range(swarmsize):
        ms = ms_init(lb, ub)
        X.append(ms)
        v.append(np.zeros_like(ms))
        p.append(np.zeros_like(ms))
    fx = np.zeros(S)  # current particle function values
    fs = np.zeros(S, dtype=bool)  # feasibility of each particle
    fp = np.ones(S) * np.inf  # best particle function values,initiated as inf
    g = np.zeros_like(lb, dtype=int)  # best swarm position
    fg = np.inf  # best swarm position value

    # Calculate objective and constraints for each particle
    if processes > 1:
        fx = np.array(mp_pool.map(obj, X))
        fs = np.array(mp_pool.map(is_feasible, X))
    else:
        for i in range(S):
            fx[i] = obj(X[i])
            fs[i] = is_feasible(X[i])

    # Store particle's best position (if constraints are satisfied)
    # attention to 'copy()', if not, bug!
    i_update = np.logical_and((fx < fp), fs)
    particle_no = 0  # particle's no (we need it to update p and fp)
    for i in i_update:
        if i:
            p[particle_no] = X[particle_no].copy()
            fp[particle_no] = fx[particle_no]
        particle_no += 1

    # Update swarm's best position
    i_min = np.argmin(fp)
    if fp[i_min] < fg:
        g = p[i_min].copy()
        fg = fp[i_min]
    else:
        # At the start, there may not be any feasible starting point, so just
        # give it a temporary "best" point since it's likely to change
        # since we didn't give a fg, so fg is still inf, it must be changed the next time
        g = X[0].copy()

    def cross(p1, p2):
        """
		cross two partial solutions
		:param p1: particle solution 1
		:param p2: particle solution 2
		:return: a better crossed solution with a less value of obj
		"""
        row, col = np.shape(p1)
        random_r = random.randint(0, col - 2)
        temp = p2.copy()
        temp[:, 0:random_r + 1] = p1[:, 0:random_r + 1].copy()
        p1[:, 0:random_r + 1] = p2[:, 0:random_r + 1].copy()
        if obj(temp) < obj(p1):
            return temp
        else:
            return p1

    def mutation(solution_p, t):
        """
        make a mutation of solution p
        :param t: annealing temperature
        :param solution_p: solution p
        :return: the mutation solution of solution p
        """
        mp = solution_p.copy()  # initialise mutation
        rows, cols = np.shape(mp)
        for row in range(rows):
            random_r = random.randint(0, cols - 1)
            high = 0
            for i_before in range(random_r + 1):
                high += mop[i_before]
            low = high - mop[random_r]
            high = high - 1
            if low != high:
                while True:
                    rand_machine = random.randint(low, high)
                    if rand_machine != mp[row, random_r]:
                        mp[row, random_r] = rand_machine
                        break
        accept_prob = random.random()
        decision_prob = math.exp(-1 * (obj(mp) - obj(solution_p)) / t)
        if decision_prob > accept_prob:
            return mp
        else:
            return solution_p

    # Initialize the particle's velocity
    # since g is given randomly, so it is feasible to initialize v as p
    v = p.copy()

    # Iterate until termination criterion met
    it = 1
    no_prove_p = np.zeros_like(fp, dtype=int)  # the generations with no prove of p
    T = 10  # initialize the annealing temperature
    a = 5  # the amplifier factor of annealing temperature
    while it <= maxiter:
        # Update the particles velocities and the particles' positions
        for i in range(S):
            v[i] = cross(v[i], cross(p[i], g))
            X[i] = cross(X[i], v[i])

        # Update objectives and constraints
        if processes > 1:
            fx = np.array(mp_pool.map(obj, X))
            fs = np.array(mp_pool.map(is_feasible, X))
        else:
            for i in range(S):
                fx[i] = obj(X[i])
                fs[i] = is_feasible(X[i])

        # Store particle's best position (if constraints are satisfied)
        i_update = np.logical_and((fx < fp), fs)
        particle_no = 0  # particle's no. (we need it to update p and fp)
        for i in i_update:
            if (not i) or fp[particle_no] - fx[particle_no] <= minfunc:
                no_prove_p[particle_no] += 1
                if no_prove_p[particle_no] >= mutation_shreshold:
                    p[particle_no] = mutation(p[particle_no], T)
                    fp[particle_no] = obj(p[particle_no])
                    no_prove_p[particle_no] = 0
            else:
                p[particle_no] = X[particle_no].copy()
                fp[particle_no] = fx[particle_no]
                no_prove_p[particle_no] = 0
            particle_no += 1

        # Compare swarm's best position with global best position
        i_min = np.argmin(fp)
        if fp[i_min] < fg:
            g = p[i_min].copy()
            fg = fp[i_min]
            if debug:
                print('New best for swarm at iteration {:}: {:} {:}'.format(it, g, fg))

        if debug:
            print('Best after iteration {:}: {:} {:}'.format(it, g, fg))
        it += 1
        T *= a

    if not is_feasible(g):
        print("However, the optimization couldn't find a feasible design. Sorry")
    if particle_output:
        return g, fg, p, fp
    else:
        return g, fg
