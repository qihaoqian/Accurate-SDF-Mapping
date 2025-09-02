import numpy as np


def numerical_hessian(x_res: float, y_res: float, f: np.ndarray):
    f2x2 = (f[1:-1, 2:] - 2 * f[1:-1, 1:-1] + f[1:-1, :-2]) / x_res**2
    f2y2 = (f[2:, 1:-1] - 2 * f[1:-1, 1:-1] + f[:-2, 1:-1]) / y_res**2
    f2xy = (f[2:, 2:] - f[2:, :-2] - f[:-2, 2:] + f[:-2, :-2]) / (4 * x_res * y_res)
    hessian = np.stack([f2x2, f2xy, f2xy, f2y2], axis=-1).reshape(
        f2x2.shape[0],
        f2x2.shape[1],
        2,
        2,
    )
    return hessian


def main():
    x_res = 1
    y_res = 1
    f = np.random.rand(5, 5)
    hessian = numerical_hessian(x_res, y_res, f)
    eigvals = np.linalg.eigvals(hessian)
    spectral_norm = np.max(np.abs(eigvals), axis=-1)
    # print(hessian)
    # print(hessian.shape)
    print(eigvals)
    print(eigvals.shape)
    print(spectral_norm)
    print(spectral_norm.shape)


if __name__ == "__main__":
    main()
