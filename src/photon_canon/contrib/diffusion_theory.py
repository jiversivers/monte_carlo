import functools
from typing import Tuple, Callable

from scipy.integrate import quad

from ..import_utils import np


def fresnel_equation(alpha_i: float = 0, n_rel: float = 1) -> float:
    """Fresnel equation for specular reflection, :ref:`FresnelEquation`,
    .. math:
        R_s = | \frac{n_1\cos\theta_i - n_2\cos\theta_t}{n_1\cos\theta_i + n_2\cos\theta_t} |
        R_p = | \frac{n_1\cos\theta_t - n_2\cos\theta_i}{n_1\cos\theta_t + n_2\cos\theta_i} |
        R = \frac{R_s + R_p}{2}
        :label: FresnelEquation

    :param alpha_i: Incident angle
    :type alpha_i: float
    :param n_rel: n2 / n1 of the interface
    :type n_rel: float
    :return: The fraction of reflected intensity
    :rtype: float
    """
    cos_alpha_t = np.sqrt(1 - np.sin(alpha_i / n_rel) ** 2)
    rs = (
        np.abs(
            (np.cos(alpha_i) - n_rel * cos_alpha_t)
            / (np.cos(alpha_i) + n_rel * cos_alpha_t)
        )
        ** 2
    )
    rp = (
        np.abs(
            (cos_alpha_t - n_rel * np.cos(alpha_i))
            / (cos_alpha_t + n_rel * np.cos(alpha_i))
        )
        ** 2
    )
    specular_reflection = 0.5 * (rs + rp)
    return specular_reflection


def cylindrical_distance(
    rtz: Tuple[float, float, float], rtz_prime: Tuple[float, float, float]
) -> float:
    """Calculate the distance between two cylindrical points, :ref:`CylindricalDistance`
    .. math:
        d = \sqrt{\left( \rho - \rho'\right) ^ 2 + \rho\rho'\cos\left( \theta - theta'\right) + \left( z - z'\right) ^ 2}
        :label: CylindricalDistance

    :param rtz: The three cylindrical coordinates (ρ, φ, z) of a point P
    :type rtz: Tuple[float, float, float]
    :param rtz_prime: The three cylindrical coordinates (ρ, φ, z) of a point P'
    :type rtz_prime: Tuple[float, float, float]
    :return: The distance between two cylindrical points
    :rtype: float
    """
    r, theta, z = rtz
    r_prime, theta_prime, z_prime = rtz_prime
    return np.sqrt(
        (r - r_prime) ** 2
        + r * r_prime * np.cos(theta - theta_prime)
        + (z - z_prime) ** 2
    )


def diffusion_approximation(
    mu_s: float,
    mu_a: float,
    r: float = 1,
    alpha_i: float = 0,
    n_rel: float = 1.33,
    g: float = 0.9,
) -> float:
    """Calculate the **steady-state diffuse reflectance** :math:`R_d(r)`
    predicted by the diffusion approximation for a semi-infinite,
    homogeneous medium illuminated at its surface.

    The formulation follows Farrell *et al.* (1992) and
    accounts for oblique illumination by adjusting the effective
    source depth and radial offset.

    Parameters
    ----------
    mu_s : float
        Scattering coefficient :math:`\mu_s\;[\text{mm}^{-1}]`.
    mu_a : float
        Absorption coefficient :math:`\mu_a\;[\text{mm}^{-1}]`.
    r : float, optional
        Radial distance on the surface where :math:`R_d` is evaluated
        (in mm). Default is ``1``.
    alpha_i : float, optional
        Angle of incidence of the source in **radians**
        (``0`` = normal incidence). Default is ``0``.
    n_rel : float, optional
        Relative refractive index :math:`n_2/n_1`
        (tissue / external medium). Default is ``1.33``.
    g : float, optional
        Scattering anisotropy factor. Default is ``0.9``.

    Returns
    -------
    float
        Diffuse reflectance :math:`R_d(r)`.

    Notes
    -----
    The diffusion approximation is valid when
    :math:`\mu_s' \gg \mu_a`, where the **reduced scattering
    coefficient** is :math:`\mu_s' = (1-g)\,\mu_s`.
    Key intermediate quantities are

    .. math::

        \mu_t' &= \mu_s' + \mu_a \\
        l'     &= \frac{1}{\mu_t'} \\
        a'     &= \frac{\mu_s'}{\mu_t'} \\[6pt]
        D      &= \frac{1}{3\left(\mu_s' + \beta\mu_a\right)}, \qquad
                 \beta = \begin{cases}
                   1 & \text{normal incidence} \\
                   0.35 & \text{oblique incidence}
                 \end{cases} \\[6pt]
        \mu_{\mathrm{eff}} &= \sqrt{\frac{\mu_a}{D}}

    The diffuse reflectance is

    .. math::

        R_d(r) \;=\; \bigl(1-R_{\mathrm{sp}}\bigr)
        \Bigl[
            a' z' \,
            \frac{\bigl(1+\mu_{\mathrm{eff}}\rho_1\bigr)
                  e^{-\mu_{\mathrm{eff}}\rho_1}}{4\pi\rho_1^{3}}
            \;+\;
            a'\!\bigl(z' + 4D\bigr)
            \frac{\bigl(1+\mu_{\mathrm{eff}}\rho_2\bigr)
                  e^{-\mu_{\mathrm{eff}}\rho_2}}{4\pi\rho_2^{3}}
        \Bigr],

    where :math:`\rho_1` and :math:`\rho_2` are the distances from the
    observation point to the real and image sources, respectively, and
    :math:`R_{\mathrm{sp}}` is the specular Fresnel reflectance.

    References
    ----------
    .. [1] **M. W. Farrell, M. S. Patterson, B. C. Wilson**
       *A diffusion-theory model of spatially resolved, steady-state
       diffuse reflectance for the noninvasive determination of tissue
       optical properties in vivo.* Medical Physics **19** (1992) 879–888.
    .. [2] **S. L. Jacques**
       *Optical properties of biological tissues: a review.*
       Physics in Medicine & Biology **58** (2013) R37–R61.
    """
    # Reduced scattering coefficient
    mu_s_prime = (1 - g) * mu_s

    # Transport coefficient
    mu_t_prime = mu_s_prime + mu_a

    # Mean free path length
    l_prime = 1 / mu_t_prime

    # Transport albedo
    a_prime = mu_s_prime / mu_t_prime

    # Azimuthal angle of source and image
    theta, theta_prime = 0, 0

    # Normal incidence
    if alpha_i == 0:
        # Diffusion coefficient
        D = 1 / (3 * (mu_s_prime + mu_a))

        # Effective attenuation coefficient
        mu_eff = np.sqrt(mu_a / D)

        # Effective source distance
        z_prime = l_prime

        # Center of diffuse reflection
        r_prime = 0

    # Oblique incidence corrections
    else:
        # Diffusion coefficient
        D = 1 / (3 * (mu_s_prime + 0.35 * mu_a))

        # Effective attenuation coefficient
        mu_eff = np.sqrt(mu_a / D)

        # Transmission angle (Snell's law)
        alpha_t = np.arcsin(np.sin(alpha_i) / n_rel)

        # Center of far diffuse reflectance
        r_prime = 3 * D * np.sin(alpha_t)

        # The new effective source distance as a result of the oblique incidence
        z_prime = r_prime / np.tan(alpha_t)

    # Fresnel reflection
    Rsp = fresnel_equation(alpha_i, n_rel)

    # Extrapolated boundary location
    z_b = -2 * D

    # Distance between observation point (r, 0, 0) and original source (0, 0, z_prime)
    rho1 = cylindrical_distance((r, theta, 0), (r_prime, theta_prime, z_prime))

    # Distance between observation point (r, 0, 0) and image source point (0, 0, -z_prime - 2*z_b)
    rho2 = cylindrical_distance(
        (r, theta, 0), (r_prime, theta_prime, -z_prime - 2 * z_b)
    )

    # Diffuse reflectance at r
    Rd = (
        a_prime
        * z_prime
        * (1 + mu_eff * rho1)
        * np.exp(-mu_eff * rho1)
        / (4 * np.pi * rho1**3)
    ) + (
        a_prime
        * (z_prime + 4 * D)
        * (1 + mu_eff * rho2)
        * np.exp(-mu_eff * rho2)
        / (4 * np.pi * rho2**3)
    )
    Rd *= 1 - Rsp
    return Rd


def create_diffusion_approximation(
    r: float = 1, alpha_i: float = 0, n_rel: float = 1.33, g: float = 0.9
) -> Callable[[float, float], float]:
    """Factory function to create a callable diffusion approximation with pre-set parameters.

    :param r: Radial distance from the source.
    :type r: float
    :param alpha_i: Incident angle of the source.
    :type alpha_i: float
    :param n_rel: Relative indices of refraction at the surface interface.
    :type n_rel: float
    :param g: Scatter anisotropy.
    :type g: float
    :return: A callable diffusion approximation
    :rtype: Callable[[float, float], float]
    """
    return functools.partial(
        diffusion_approximation, r=r, alpha_i=alpha_i, n_rel=n_rel, g=g
    )


def create_integrated_diffusion_approximation(
    r_range: Tuple[float, float],
    alpha_i: float = 0,
    n_rel: float = 1.33,
    g: float = 0.9,
) -> Callable[[float, float], float]:
    """
    Build a callable that returns the **radially integrated diffuse
    reflectance**

    .. math::

        \tilde{R}_d(\mu_s, \mu_a)
        \;=\;
        2\pi \int_{r_{\min}}^{r_{\max}}
        r \, R_d\!\bigl(r;\mu_s,\mu_a\bigr)\;dr,

    where :math:`R_d(r;\mu_s,\mu_a)` is the spatially resolved
    reflectance given by :pyfunc:`diffusion_approximation`.

    Parameters
    ----------
    r_range : (float, float)
        Integration bounds :math:`(r_{\min}, r_{\max})` in millimetres.
    alpha_i : float, optional
        Angle of incidence of the illumination in **radians**
        (``0`` ⇒ normal incidence). Default ``0``.
    n_rel : float, optional
        Relative refractive index :math:`n_2/n_1`
        (tissue / external medium). Default ``1.33``.
    g : float, optional
        Scattering anisotropy factor. Default ``0.9``.

    Returns
    -------
    Callable[[float, float], float]
        A function ``f(mu_s, mu_a)`` that evaluates
        :math:`\tilde{R}_d(\mu_s,\mu_a)` using
        :pyfunc:`scipy.integrate.quad`.

    Notes
    -----
    The integrand is

    .. math::

        f(r,\mu_s,\mu_a)
        \;=\;
        2\pi\,r\,R_d(r;\mu_s,\mu_a),

    so the returned function effectively computes

    .. math::

        \tilde{R}_d(\mu_s,\mu_a)
        \;=\;
        \int_{r_{\min}}^{r_{\max}} f(r,\mu_s,\mu_a)\;dr.

    The numerical integration is performed with adaptive quadrature
    (:pyfunc:`scipy.integrate.quad`).

    See Also
    --------
    diffusion_approximation : Point-wise diffusion reflectance model.

    Examples
    --------
    >>> integ_Rd = create_integrated_diffusion_approximation((0.0, 5.0))
    >>> integ_Rd(mu_s=10.0, mu_a=0.1)
    0.0487  # doctest: +SKIP
    """

    def integrand(r: float, mu_s: float, mu_a: float) -> float:
        return (
            2
            * np.pi
            * r
            * diffusion_approximation(
                mu_s=mu_s, mu_a=mu_a, r=r, alpha_i=alpha_i, n_rel=n_rel, g=g
            )
        )

    def integrated_diffusion_approximation(mu_s: float, mu_a: float) -> float:
        result, err = quad(integrand, r_range[0], r_range[1], args=(mu_s, mu_a))
        return result

    return integrated_diffusion_approximation
