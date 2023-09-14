"""Time stepping (integration) tools."""

import numpy as np
import scipy.linalg as sla
from IPython.lib.pretty import pretty as pretty_repr

from dapper.tools.progressbar import progbar
from dapper.tools.seeding import rng

from .utils import NamedFunc


# fmt: off
def rk4(f, x, t, dt, stages=4, s=0):
    """Runge-Kutta (explicit, non-adaptive) numerical (S)ODE solvers.

    For ODEs, the order of convergence equals the number of `stages`.

    For SDEs with additive noise (`s>0`), the order of convergence
    (both weak and strong) is 1 for `stages` equal to 1 or 4.
    These correspond to the classic Euler-Maruyama scheme and the Runge-Kutta
    scheme for S-ODEs respectively, see `bib.grudzien2020numerical`
    for a DA-specific discussion on integration schemes and their discretization errors.

    Parameters
    ----------
    f : function
        The time derivative of the dynamical system. Must be of the form `f(t, x)`

    x : ndarray or float
        State vector of the forcing term

    t : float
        Starting time of the integration

    dt : float
        Integration time step.

    stages : int, optional
        The number of stages of the RK method.
        When `stages=1`, this becomes the Euler (-Maruyama) scheme.
        Default: 4.

    s : float
        The diffusion coeffient (std. dev) for models with additive noise.
        Default: 0, yielding deterministic integration.

    Returns
    -------
    ndarray
        State vector at the new time, `t+dt`
    """

    # Draw noise
    if s > 0:
        W = s * np.sqrt(dt) * rng.standard_normal(x.shape)
    else:
        W = 0

    # Approximations to Delta x
    if stages >= 1: k1 = dt * f(x,           t)         + W    # noqa
    if stages >= 2: k2 = dt * f(x+k1/2.0,    t+dt/2.0)  + W    # noqa
    if stages == 3: k3 = dt * f(x+k2*2.0-k1, t+dt)      + W    # noqa
    if stages == 4:
                    k3 = dt * f(x+k2/2.0,    t+dt/2.0)  + W    # noqa
                    k4 = dt * f(x+k3,        t+dt)      + W    # noqa

    # Mix proxies
    if    stages == 1: y = x + k1                              # noqa
    elif  stages == 2: y = x + k2                              # noqa
    elif  stages == 3: y = x + (k1 + 4.0*k2 + k3)/6.0          # noqa
    elif  stages == 4: y = x + (k1 + 2.0*(k2 + k3) + k4)/6.0   # noqa
    else:
        raise NotImplementedError

    return y


def with_rk4(dxdt, autonom=False, stages=4, s=0):
    """Wrap `dxdt` in `rk4`."""
    def tendency(x, t):
        if autonom:
            return dxdt(x)
        else:
            return dxdt(x, t)

    def step(x0, t0, dt):
        return rk4(tendency, x0, t0, dt, stages=stages)

    name = "rk"+str(stages)+" integration of "+pretty_repr(dxdt)
    step = NamedFunc(step, name)
    return step


def with_recursion(func, prog=False):
    """Make function recursive in its 1st arg.

    Return a version of `func` whose 2nd argument (`k`)
    specifies the number of times to times apply func on its output.

    .. warning:: Only the first argument to `func` will change,
        so, for example, if `func` is `step(x, t, dt)`,
        it will get fed the same `t` and `dt` at each iteration.

    Parameters
    ----------
    func : function
        Function to recurse with.

    prog : bool or str
        Enable/Disable progressbar. If `str`, set its name to this.

    Returns
    -------
    fun_k : function
        A function that returns the sequence generated by recursively
        running `func`, i.e. the trajectory of system's evolution.
    Examples
    --------
    >>> def dxdt(x):
    ...     return -x
    >>> step_1  = with_rk4(dxdt, autonom=True)
    >>> step_k  = with_recursion(step_1)
    >>> x0      = np.arange(3)
    >>> x7      = step_k(x0, 7, t0=np.nan, dt=0.1)[-1]
    >>> x7_true = x0 * np.exp(-0.7)
    >>> np.allclose(x7, x7_true)
    True
    """
    def fun_k(x0, k, *args, **kwargs):
        xx = np.zeros((k+1,)+x0.shape)
        xx[0] = x0

        # Prog. bar name
        if prog == False:
            desc = None
        elif prog == None:
            desc = "Recurs."
        else:
            desc = prog

        for i in progbar(range(k), desc):
            xx[i+1] = func(xx[i], *args, **kwargs)

        return xx

    return fun_k


def integrate_TLM(TLM, dt, method='approx'):
    r"""Compute the resolvent.

    The resolvent may also be called

    - the Jacobian of the step func.
    - the integral of (with *M* as the TLM):
      $$ \frac{d U}{d t} = M U, \quad U_0 = I .$$

    .. note:: the tangent linear model (TLM)
              is assumed constant (for each `method` below).

    Parameters
    ----------
    method : str

        - `'approx'`  : derived from the forward-euler scheme.
        - `'rk4'`     : higher-precision approx.
        - `'analytic'`: exact.
        .. warning:: 'analytic' typically requries higher inflation in the ExtKF.

    See Also
    --------
    `FD_Jac`.
    """
    if method == 'analytic':
        Lambda, V = sla.eig(TLM)
        resolvent = (V * np.exp(dt*Lambda)) @ np.linalg.inv(V)
        resolvent = np.real_if_close(resolvent, tol=10**5)
    else:
        Id = np.eye(TLM.shape[0])
        if method == 'rk4':
            resolvent = rk4(lambda U, t: TLM@U, Id, np.nan, dt)
        elif method.lower().startswith('approx'):
            resolvent = Id + dt*TLM
        else:
            raise ValueError
    return resolvent


def FD_Jac(func, eps=1e-7):
    """Finite-diff approx. of Jacobian of `func`.

    The function `func(x)` must be compatible with `x.ndim == 1` and `2`,
    where, in the 2D case, each row is seen as one function input.

    Returns
    -------
    function
        The first input argument is that of which the derivative is taken.


    Examples
    --------
    >>> dstep_dx = FD_Jac(step) # doctest: +SKIP
    """
    def Jac(x, *args, **kwargs):
        def f(y):
            return func(y, *args, **kwargs)
        E = x + eps*np.eye(len(x))  # row-oriented ensemble
        FT = (f(E) - f(x))/eps      # => correct broadcasting
        return FT.T                 # => Jac[i,j] = df_i/dx_j
    return Jac

# Transpose explanation:
# - Let F[i,j] = df_i/dx_j be the Jacobian matrix such that
#               f(A)-f(x) ≈ F @ (A-x)
#   for a matrix A whose columns are realizations. Then
#                       F ≈ [f(A)-f(x)] @ inv(A-x)   [eq1]
# - But, to facilitate broadcasting,
#   DAPPER works with row-oriented (i.e. "ens_compatible" functions),
#   meaning that f should be called as f(A').
#   Transposing [eq1] yields:
#        F' = inv(A-x)'  @ [f(A)  - f(x)]'
#           = inv(A'-x') @ [f(A') - f(x')]
#           =      1/eps * [f(A') - f(x')]
#           =              [f(A') - f(x')] / eps     [eq2]
# => Need to compute [eq2] and then transpose.
