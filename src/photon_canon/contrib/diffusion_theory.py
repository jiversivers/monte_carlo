import functools
from typing import Tuple, Callable

from scipy.integrate import quad

from ..import_utils import np

# TODO: Check that everything in these functions will vectorize


def fresnel_equation(alpha_i: float = 0, n_rel: float = 1) -> float:
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
    return functools.partial(
        diffusion_approximation, r=r, alpha_i=alpha_i, n_rel=n_rel, g=g
    )


def create_integrated_diffusion_approximation(
    r_range: Tuple[float, float],
    alpha_i: float = 0,
    n_rel: float = 1.33,
    g: float = 0.9,
) -> Callable[[float, float], float]:
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
