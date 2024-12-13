<!-- PROJECT SHIELDS -->
<!-- [![arXiv][arxiv-shield]][arxiv-url] -->
[![DOI](https://zenodo.org/badge/902835677.svg)](https://doi.org/10.5281/zenodo.14443259)
<!--[![MIT License][license-shield]][license-url]-->

# Certified Model Predictive Control For Switched Evolution Equations Using Model Order Reduction

In this repository we provide the code for the paper "Certified Model Predictive Control For Switched Evolution Equations Using Model Order Reduction" by Michael Kartmann, Mattia Manucci, Benjamin Unger, and Stefan Volkwein.

## Installation
A python environment is required with at least **Python 3.10.12**.

Install dependencies via `pip`:
   ```sh
   pip install -r requirements.txt
   ```

<!-- USAGE EXAMPLES -->
## Organization of the repository

The repository contains the directory [`Code/`](), which includes the source code. The code consists of the main files

* `main_mpc.py`: the main file comparing the performance of the MPC schemes,
* `main_openloop_error_estimation.py`: the main file testing the error estimators of the open-loop problems,
* `main_state_adjoint_error_estimation.py`: the main file for testing the error estimators of the state and adjoint equation,
* `main_pretrained_mpc_error_estimation.py`: the main file for testing the recursive MPC error estimators.

Further, we provide the following files to handle the discretization and model reduction:

* `discretizre.py`: contains all routines to obtain the full-order model (FOM) by Finite Element Discretization,
* `model.py`: contains the implementation of the full-order or reduced-order model (ROM),
* `reductor.py`: contains all reduction routines,
* `mpc.py`: contains the implementation of the mpc schemes.

<!-- LICENSE -->
## License

Distributed under the MIT License. See [`LICENSE`][license-url] for more information.

<!--  ## Citing
If you use this project for academic work, please consider citing our
[work][arxiv-url]:

    M. Kartmann, M. Manucci, B. Unger, S. Volkwein
    Certified Model Predictive Control for Switched Evolution Equations using Model Order Reduction
--> 
<!-- CONTACT -->
## Contact
Michael Kartmann - michael.kartmanns@uni-konstanz.de

[license-url]: https://github.com/michikartmann/test/blob/main/LICENSE
<!--[doi-shield]: https://img.shields.io/badge/DOI-10.5281%20%2F%20zenodo.6497497-blue.svg?style=for-the-badge
<!-- [doi-url]:
[arxiv-shield]: https://img.shields.io/badge/arXiv-2204.13474-b31b1b.svg?style=for-the-badge
[arxiv-url]:  -->
