# Literature Survey: Compressed Sensing & Sparse Spectral Estimation for Ocean Wave Field Reconstruction

Date: 2026-05-21
Context: WaveSplat project -- exploring CS/sparse methods to increase spectral resolution from K=20 to K=100+

---

## Table of Contents

1. [Foundational CS / Sparse Recovery Theory](#1-foundational-cs--sparse-recovery-theory)
2. [Compressed Sensing Directly Applied to Ocean Waves](#2-compressed-sensing-directly-applied-to-ocean-waves)
3. [Sparse Spectral Estimation (Gridless / Atomic Norm)](#3-sparse-spectral-estimation-gridless--atomic-norm)
4. [Sparse Wavenumber Recovery (SHM / Acoustics -- Transferable Ideas)](#4-sparse-wavenumber-recovery-shm--acoustics----transferable-ideas)
5. [Wave Field Reconstruction from Sparse Measurements (Non-CS)](#5-wave-field-reconstruction-from-sparse-measurements-non-cs)
6. [Data Assimilation / Kalman Filter Approaches](#6-data-assimilation--kalman-filter-approaches)
7. [Deep Learning for Sparse Wave Reconstruction](#7-deep-learning-for-sparse-wave-reconstruction)
8. [Grid-Based Spectral Estimation for Waves](#8-grid-based-spectral-estimation-for-waves)
9. [Summary Table](#9-summary-table)
10. [Relevance to WaveSplat](#10-relevance-to-wavesplat)

---

## 1. Foundational CS / Sparse Recovery Theory

### 1.1 Candes, Romberg, Tao (2006) -- Seminal CS Papers

- **Title**: "Robust Uncertainty Principles: Exact Signal Reconstruction from Highly Incomplete Frequency Information"
- **Authors**: E. Candes, J. Romberg, T. Tao
- **Venue**: IEEE Trans. Information Theory, 52(2), 2006
- **Also**: "Stable Signal Recovery from Incomplete and Inaccurate Measurements", Comm. Pure Appl. Math., 59(8), 2006
- **Method**: L1 minimization / basis pursuit
- **Key result**: Sparse signals can be exactly recovered from far fewer measurements than Nyquist requires, via convex optimization (L1 norm minimization)
- **Relevance**: Theoretical foundation for all CS-based wave reconstruction. If the wave field is sparse in wavenumber domain (K active components out of N grid points), CS guarantees recovery with O(K log N) measurements.
- **Code**: N/A (foundational theory)

### 1.2 Candes & Fernandez-Granda (2014) -- Super-Resolution Theory

- **Title**: "Towards a Mathematical Theory of Super-Resolution"
- **Authors**: E. Candes, C. Fernandez-Granda
- **Venue**: Comm. Pure Appl. Math., 67(6), 2014
- **Method**: Total variation norm minimization on measures (continuous analog of L1)
- **Key result**: Continuous sparse frequency spectra can be recovered with infinite precision via convex optimization, given sufficient separation between spectral lines. No grid discretization needed.
- **Relevance**: Directly applicable to our problem -- wave field as sum of spectral components with continuous (kx, ky) values. Resolves grid mismatch issues.
- **Code**: N/A (theory)

### 1.3 Tang, Bhaskar, Recht (2013) -- Compressed Sensing Off the Grid

- **Title**: "Compressed Sensing Off the Grid"
- **Authors**: G. Tang, B. Bhaskar, B. Recht
- **Venue**: IEEE Trans. Information Theory, 59(11), 2013
- **Method**: Atomic norm minimization (ANM) -- convex relaxation for continuous dictionary CS
- **Key result**: Extends CS to continuous parameter space (no grid needed). Recovery via semidefinite programming (SDP). Near-minimax optimal for line spectral estimation.
- **Relevance**: Core method for gridless sparse spectral estimation. Could replace our discrete K-component model with a continuous spectral representation.
- **Code**: N/A

---

## 2. Compressed Sensing Directly Applied to Ocean Waves

### 2.1 Bayindir (2016) -- Compressive Spectral Method for Nonlinear Gravity Waves

- **Title**: "Compressive Spectral Method for the Simulation of the Nonlinear Gravity Waves"
- **Authors**: C. Bayindir
- **Venue**: Scientific Reports, 6, 22100 (2016)
- **Method**: Compressive sampling (CS) combined with higher-order spectral (HOS) simulation
- **Input**: Simulated wave fields (1D)
- **Waves**: 1D, fully nonlinear
- **Key result**: Ocean waves are sparse in frequency domain. CS reduces number of spectral components needed for HOS simulation by ~90% with negligible accuracy loss. 4th-order Runge-Kutta time integration with compressive spectral evaluation.
- **Limitations**: Forward simulation only (not reconstruction from observations). 1D waves.
- **Code**: Not available
- **URL**: https://www.nature.com/articles/srep22100

### 2.2 Bayindir & Nayfeh (2018) -- Extrapolation of Random Wave Field via CS

- **Title**: "Extrapolation of Random Wave Field Data via Compressive Sampling"
- **Authors**: C. Bayindir (+ collaborators)
- **Venue**: Ocean Engineering (2018)
- **Method**: L1-norm minimization with directional spectrum basis
- **Input**: Simulated random wave field data at sparse spatial points
- **Waves**: 2D directional sea states
- **Key result**: Wave field can be reconstructed/extrapolated from sparse spatial data using CS. Main features of frequency spectra and spatial cross-correlations are preserved even with >60% missing data.
- **Limitations**: Relies on known directional spectrum model as sparsifying basis. Tested on synthetic data only.
- **Code**: Not available

### 2.3 Lyu et al. (2025) -- Nonneg. Sparse Regularization for HF Radar Wave Spectrum

- **Title**: "Ocean Wave Spectrum Reconstruction from HF Radar Data and Its Application to Wave Height Estimation"
- **Authors**: Lyu et al.
- **Venue**: arXiv:2505.13132 (2025)
- **Method**: Nonnegative sparse regularization (Tikhonov functional with L1 + nonnegativity constraints)
- **Input**: HF radar second-order Doppler spectrum
- **Waves**: 1D (nondirectional spectrum recovery)
- **Key result**: Recovers ocean wave spectrum from HF radar backscatter using inverse problem formulation with sparse regularization. Stability estimate proved. Validated on simulations.
- **Limitations**: Recovers energy spectrum (not phase-resolved). Specific to HF radar measurement model.
- **Code**: Not available
- **URL**: https://arxiv.org/pdf/2505.13132

### 2.4 Laface et al. (2017, 2022) -- CS for Sea Surface Elevation Reconstruction

- **Title**: "Compressive Sensing-Based Reconstruction of Sea Free-Surface Displacement" (2017); "Compressive Sampling-Based Extrapolation of Free Surface Displacement Data from Pressure Measurements" (2022)
- **Authors**: V. Laface, A. Romolo, F. Arena et al.
- **Venue**: Various (2017, Ocean Engineering 2022)
- **Method**: CS with harmonic wavelet basis for reconstruction of free surface records
- **Input**: Pressure sensor measurements / partial time-series data
- **Waves**: 1D (time-series reconstruction)
- **Key result**: Reliable reconstruction of sea surface elevation records with >60% missing data. Extended to pressure-to-surface inversion in 2022 with analysis of steep wave limitations.
- **Limitations**: Temporal reconstruction only (not spatial field). Steep waves degrade performance.
- **Code**: Not available

---

## 3. Sparse Spectral Estimation (Gridless / Atomic Norm)

### 3.1 Atomic Norm Minimization for Line Spectral Estimation

- **Title**: "Atomic Norm Denoising With Applications to Line Spectral Estimation"
- **Authors**: B. Bhaskar, G. Tang, B. Recht
- **Venue**: IEEE Trans. Information Theory, 59(6), 2013
- **Method**: Atomic norm minimization (ANM) via semidefinite programming (SDP)
- **Key result**: Recovers continuous-valued frequencies from noisy measurements without discretizing frequency domain. Achieves super-resolution. Equivalent to finding a positive semidefinite Toeplitz matrix (Caratheodory-Fejer decomposition).
- **Relevance**: Directly applicable to our (kx, ky) estimation. Could replace discrete grid search with continuous optimization.
- **Limitation**: Computational cost O(N^3) for SDP. 2D extension (decoupled ANM) exists but scales poorly.
- **Code**: Various implementations exist (CVX/CVXPY based)

### 3.2 Decoupled Atomic Norm Minimization for 2D Line Spectral Estimation

- **Title**: "Efficient Two-Dimensional Line Spectrum Estimation Based on Decoupled Atomic Norm Minimization"
- **Authors**: Y. Yang, Z. Chu
- **Venue**: Signal Processing, 2019
- **Method**: Decoupled ANM -- solves 2D frequency estimation by decoupling into separate dimensions
- **Key result**: Reduces computational complexity of 2D ANM from intractable to feasible. Handles 2D frequency pairs (analogous to kx, ky).
- **Relevance**: Directly relevant for 2D wavenumber estimation. However, still requires SDP solver.
- **Code**: Not publicly available

### 3.3 ADMM for ND Line Spectral Estimation (Grid-Free CS)

- **Title**: "ADMM for ND Line Spectral Estimation using Grid-Free Compressive Sensing from Multiple Measurements with Applications to DOA Estimation"
- **Authors**: Y. Yang et al.
- **Venue**: arXiv:1811.02477 (2018)
- **Method**: ADMM-based solver for multidimensional atomic norm minimization
- **Key result**: Scalable algorithm for gridless CS in N dimensions. Applied to DOA estimation.
- **Relevance**: Could handle joint (kx, ky) estimation without grid. ADMM more scalable than interior-point SDP solvers.
- **Code**: Not publicly available

### 3.4 Sparse Bayesian Learning (SBL) for Spectral Estimation

- **Title**: "Pattern-Coupled Sparse Bayesian Learning for Recovery of Block-Sparse Signals"
- **Authors**: Various (Fang, Wang et al., IEEE TSP 2016 is a key reference)
- **Venue**: IEEE Trans. Signal Processing, 64(18), 2016
- **Method**: Sparse Bayesian Learning / Relevance Vector Machine with hierarchical priors
- **Key result**: Bayesian approach automatically determines sparsity level. Can achieve super-resolution. More robust than L1 methods to dictionary coherence.
- **Relevance**: Could be used for wavenumber spectrum estimation with automatic model order selection (no need to pre-specify K).
- **Limitation**: Iterative, slower than single-shot L1. Grid-based unless combined with off-grid extensions.
- **Code**: Various implementations available

---

## 4. Sparse Wavenumber Recovery (SHM / Acoustics -- Transferable Ideas)

### 4.1 Harley & Moura (2013) -- Sparse Wavenumber Recovery for Lamb Waves

- **Title**: "Sparse Recovery of the Multimodal and Dispersive Characteristics of Lamb Waves"
- **Authors**: J.B. Harley, J.M.F. Moura
- **Venue**: J. Acoustical Society of America, 133(5), 2732-2745 (2013)
- **Method**: L1 optimization (basis pursuit) on a wavenumber dictionary constrained by dispersion relation
- **Input**: Surface-mounted transducer array (sparse sensors)
- **Waves**: Multimodal dispersive Lamb waves in plates
- **Key result**: Accurately recovers frequency-wavenumber representation from limited sensors (10-20% of spatial data). Outperforms MUSIC and FFT in angular resolution and noise robustness. Dispersion relation used to structure the dictionary.
- **Relevance**: **HIGHLY RELEVANT** -- same mathematical structure as our problem. Sparse sensors observing dispersive waves, recovery of wavenumber spectrum via L1 with physics (dispersion) constraint. Directly transferable approach.
- **Code**: Available (CodeOcean repository: https://smartdata.ece.ufl.edu)

### 4.2 Hu et al. (2022) -- Hybrid LASSO for Guided Wave Wavenumber Analysis

- **Title**: "Sparse Wavenumber Analysis of Guided Wave Based on Hybrid Lasso Regression in Composite Laminates"
- **Authors**: Y. Hu, F. Cui, F. Li, X. Tu, L. Zeng
- **Venue**: Structural Health Monitoring, 2022
- **Method**: Hybrid LASSO with group-sparsity and continuity regularization
- **Input**: Sparse sensor array on composite plate
- **Waves**: Guided waves in composite laminates (dispersive, anisotropic)
- **Key result**: Group-sparsity constraint improves resolution over standard LASSO. Continuity regularization enforces smooth dispersion curves. Better than FFT and standard SWA.
- **Relevance**: Group-LASSO idea transferable -- wave components at nearby (kx, ky) could be grouped.
- **Code**: Not available

### 4.3 Le Courtois & Bonnel (2015) -- CS for Wideband Wavenumber Tracking

- **Title**: "Compressed Sensing for Wideband Wavenumber Tracking in Dispersive Shallow Water"
- **Authors**: F. Le Courtois, J. Bonnel
- **Venue**: J. Acoustical Society of America, 138(2), 575-583 (2015)
- **Method**: CS sparse recovery + particle filter tracking across frequencies
- **Input**: 32-sensor horizontal line array (SHARK array)
- **Waves**: Acoustic modes in shallow water (dispersive)
- **Key result**: CS recovers wavenumbers at each frequency; particle filter tracks them across frequency bands using dispersion relation as dynamic model. Applied to real Shallow Water 2006 data.
- **Relevance**: **RELEVANT** -- demonstrates CS + dispersion tracking for real dispersive wave data. The "track wavenumber across frequency" idea parallels our problem of finding spectral components that satisfy the dispersion relation.
- **Code**: Not available

---

## 5. Wave Field Reconstruction from Sparse Measurements (Non-CS)

### 5.1 Desmars, Ducrozet, Grilli et al. (2018, 2023) -- Phase-Resolved Reconstruction from Optical Data

- **Title**: "Phase-Resolved Reconstruction Algorithm and Deterministic Prediction of Nonlinear Ocean Waves From Spatio-Temporal Optical Measurements" (OMAE 2018); "Nonlinear Deterministic Reconstruction and Prediction of Remotely Measured Ocean Surface Waves" (JFM, 2023)
- **Authors**: N. Desmars, Y. Pérignon, G. Ducrozet, C.A. Guérin, S.T. Grilli, P. Ferrant
- **Venue**: ASME OMAE 2018; J. Fluid Mechanics, 975 (2023)
- **Method**: Spectral decomposition + HOS nonlinear correction. Linear wave theory for initial reconstruction, then 2nd-order Stokes correction.
- **Input**: Remote optical sensor (LiDAR, stereo-video) on fixed platform -- spatiotemporal surface elevation fields
- **Waves**: 2D, nonlinear (up to 2nd order)
- **Key result**: Accurate phase-resolved reconstruction and prediction. 2023 paper shows HOS-based nonlinear model significantly improves over linear for steep waves and long prediction horizons.
- **Limitations**: Requires spatiotemporally dense optical measurements (not truly sparse point observations).
- **Code**: Not available

### 5.2 Gallego, Yezzi et al. (2021) -- Real-Time Optical Wave Estimation via Dispersion

- **Title**: "Toward Real-Time Optical Estimation of Ocean Waves' Space-Time Fields"
- **Authors**: (Benetazzo, Bergamasco, Gallego, Yezzi et al.)
- **Venue**: Computers & Geosciences, 147, 104666 (2021)
- **Method**: Sparse stereo 3D point matching + NUFFT spectrum estimation + dispersion-based prediction/update cycle
- **Input**: Stereo camera pair -- sparse 3D feature matches between frames
- **Waves**: 2D surface gravity waves (linear dispersion assumed)
- **Key result**: >10x faster than dense stereo (WASS). Uses linear dispersion relation to propagate spectrum between frames and update with new sparse observations. Real-time capable.
- **Relevance**: **HIGHLY RELEVANT** -- same idea of using sparse 3D points + dispersion relation to estimate wave spectrum. Key difference: they use temporal coherence (frame-to-frame), we use single-snapshot.
- **Code**: WASS pipeline is open-source (https://github.com/fbergama/wass), but the fast spectral method described in this paper is not separately released.

### 5.3 LiDAR-Based Wave Spectral Analysis

- **Title**: "Retrieval of Directional Power Spectral Density and Wave Parameters from Airborne LiDAR Point Cloud" (Ocean Engineering, 2022); "Wave Characteristics Analysis by LiDAR Point Cloud" (Ocean Engineering, 2025)
- **Authors**: Various groups
- **Venue**: Ocean Engineering (2022, 2025)
- **Method**: Welch method / EMEP (Extended Maximum Entropy Principle) on LiDAR point clouds
- **Input**: Drone-mounted or airborne LiDAR -- dense point clouds
- **Waves**: 2D directional spectra
- **Key result**: Demonstrated retrieval of directional spectra from LiDAR data. EMEP resolves multi-peak directional distributions.
- **Limitations**: Assumes dense spatial coverage (airborne). Not designed for ship-mounted sparse observations.
- **Code**: Not available

---

## 6. Data Assimilation / Kalman Filter Approaches

### 6.1 Wang & Pan (2021, 2022) -- EnKF-HOS for Wave Forecast

- **Title**: "Phase-Resolved Ocean Wave Forecast with Ensemble-Based Data Assimilation" (JFM, 2021); "Phase-Resolved Ocean Wave Forecast with Simultaneous Current Estimation through Data Assimilation" (JFM, 2022)
- **Authors**: G. Wang, Y. Pan
- **Venue**: J. Fluid Mechanics (2021, 2022)
- **Method**: Ensemble Kalman Filter (EnKF) coupled with Higher-Order Spectral (HOS) method
- **Input**: Sparse wave probes or radar-like observations
- **Waves**: 2D, nonlinear (HOS up to 3rd/4th order)
- **Key result**: First EnKF-based data assimilation for phase-resolved wave reconstruction and forecast. State vector contains spectral coefficients on a grid. 2022 extension simultaneously estimates ocean current.
- **Relevance**: **RELEVANT** -- state vector is spectral coefficients (like our model). EnKF handles uncertainty. But grid-based spectrum (fixed resolution) and computationally heavy (ensemble of HOS runs).
- **Code**: GitHub repo exists (fengHEU/Wave-ETKF) but core ETKF code not publicly released (copyright pending).

### 6.2 Feng Wang et al. (2024) -- ETKF for Wave Field from Optical Points

- **Title**: "From Points to Waves: Fast Ocean Wave Spatial-Temporal Fields Estimation Using Ensemble Transform Kalman Filter with Optical Measurement"
- **Authors**: F. Wang, Q. Zhu, C. Cai, X. Wang, R. Qiao
- **Venue**: Coastal Engineering, 2024
- **Method**: Ensemble Transform Kalman Filter (ETKF) with linear dispersion prediction model
- **Input**: Sparse 3D points from optical sensor (stereo camera / LiDAR), interpolated to grid
- **Waves**: 2D surface waves (linear dispersion)
- **Key result**: Fast recursive estimation using ETKF. Sparse 3D points interpolated to grid as measurement. Fuzzy logic for outlier handling. Superior efficiency and accuracy on public + field datasets. MAP estimation inherently achieves global optimization.
- **Relevance**: **DIRECTLY COMPARABLE** to WaveSplat -- same problem (sparse 3D points to wave field). Different method (ETKF vs spectral Gaussian optimization). Already implemented as baseline in WaveSplat (nRMSE 0.319 vs WaveSplat 0.223).
- **Code**: Not fully public (fengHEU/Wave-ETKF partial)

---

## 7. Deep Learning for Sparse Wave Reconstruction

### 7.1 Ehlers, Klein et al. (2023) -- ML for Phase-Resolved Wave Reconstruction

- **Title**: "Machine Learning for Phase-Resolved Reconstruction of Nonlinear Ocean Wave Surface Elevations from Sparse Remote Sensing Data"
- **Authors**: S. Ehlers, M. Klein, A. Heinlein, M. Wedler, N. Desmars, N. Hoffmann, M. Stender
- **Venue**: Ocean Engineering, 288, 116059 (2023)
- **Method**: U-Net and Fourier Neural Operator (FNO) trained on HOS-simulated wave data with geometric radar model
- **Input**: Synthetic radar snapshots (spatiotemporal sparse data)
- **Waves**: 1D nonlinear (HOS-generated)
- **Key result**: FNO outperforms U-Net due to Fourier-space learning naturally suited to wave structure. Spatiotemporal input (multiple historic snapshots) greatly improves accuracy over single snapshot.
- **Limitations**: Trained on specific sea state ranges. Generalization to 2D not demonstrated. Requires training data.
- **Code**: Not available

### 7.2 Ehlers et al. (2025) -- Physics-Informed Neural Operator (PINO)

- **Title**: "Bridging Ocean Wave Physics and Deep Learning: Physics-Informed Neural Operators for Nonlinear Wavefield Reconstruction in Real-Time"
- **Authors**: S. Ehlers et al.
- **Venue**: Physics of Fluids, 37(10), 107119 (2025)
- **Method**: Physics-Informed Neural Operator (PINO) with free surface boundary condition residuals in loss
- **Input**: Buoy time series or radar snapshots (sparse)
- **Waves**: 2D nonlinear
- **Key result**: No ground truth needed during training (physics loss only). Once trained, inference <1 second per wave instance. Generalizes across wave conditions. Handles indirect measurements (radar backscatter).
- **Relevance**: Alternative paradigm -- instead of explicit spectral decomposition, learn the mapping from sparse observations to full field. Real-time capable.
- **Code**: Not available

### 7.3 SHRED for Wave Height Reconstruction (2025)

- **Title**: "Sparse-to-Full Reconstruction of Ocean Wave Height Fields Using Lightweight Recurrent Neural Models"
- **Authors**: (Various)
- **Venue**: Ocean Engineering (2025)
- **Method**: SHRED (SHallow REcurrent Decoder) = LSTM + shallow decoder
- **Input**: 1-2 buoy time series (extremely sparse)
- **Waves**: Significant wave height (SWH) fields (not phase-resolved)
- **Key result**: MRE < 0.085, Nash-Sutcliffe > 0.95 from just 1-2 buoys. 67% MRE reduction over standard shallow decoder. Uses temporal dynamics to compensate for spatial sparsity.
- **Limitations**: SWH fields only (not phase-resolved). Relies on statistical patterns, not physics.
- **Code**: Not available

### 7.4 Lv et al. (2025) -- Framework for Observation Point Selection & Reconstruction

- **Title**: "A New Framework for Selecting Observation Points and Reconstructing Wave Fields Under Sparse Observations"
- **Authors**: Lv et al.
- **Venue**: Coastal Engineering (2025)
- **Method**: Optimal sensor placement + deep learning reconstruction
- **Input**: Sparse buoy observations (ERA5 reanalysis data)
- **Waves**: SWH fields (South China Sea)
- **Key result**: Optimized observation point placement significantly outperforms conventional deployment strategies for reconstruction accuracy.
- **Code**: Not available

---

## 8. Grid-Based Spectral Estimation for Waves

### 8.1 X-Band Radar 3D-FFT Methods (Classical)

- **Title**: Various (Nieto Borge et al., 2004; Dankert et al., 2005)
- **Method**: 3D FFT of radar image sequence -> wavenumber-frequency spectrum -> filter by dispersion relation -> invert
- **Input**: X-band marine radar image sequences (dense spatial, temporal)
- **Waves**: 2D directional
- **Key result**: Standard operational method. Uses dispersion shell filtering to extract wave signal from noise. Requires calibration (MTF, shadowing correction).
- **Limitations**: Grid-based FFT resolution limited by image size. Requires dense spatial data (radar footprint). Not applicable to sparse point observations.
- **Code**: Various commercial implementations

### 8.2 POD-Based Wave Field Extraction from Radar

- **Title**: "Use of Proper Orthogonal Decomposition for Extraction of Ocean Surface Wave Fields from X-Band Radar Measurements"
- **Authors**: (Dankert et al.)
- **Venue**: Remote Sensing, 9(9), 881 (2017)
- **Method**: POD/EOF decomposition of radar image sequences
- **Input**: X-band radar images
- **Waves**: 2D
- **Key result**: POD captures dominant wave system accurately. Alternative to FFT-based methods.
- **Code**: Not available

### 8.3 Iterative Threshold Shrinkage for High-Resolution Dispersion Spectra

- **Title**: "High-Resolution Surface Wave Dispersion Spectrum Computation Based on Iterative Threshold Shrinkage Algorithm and Its Application to Irregularly Sampled Data"
- **Authors**: Various
- **Venue**: Frontiers in Earth Science (2025)
- **Method**: Sparse inversion (iterative thresholding / Tau-P transform) for dispersion spectrum from irregular samples
- **Input**: Irregularly sampled spatial data (seismic, but method is general)
- **Waves**: Surface waves (seismic context)
- **Key result**: High-resolution frequency-phase velocity dispersion spectrum from irregular spatial sampling. Outperforms traditional MASW methods.
- **Relevance**: Demonstrates sparse inversion for dispersion spectrum from irregular samples -- exactly our scenario (irregular LiDAR points).
- **Code**: Not available

---

## 9. Summary Table

| Paper | Year | Method | Input Type | 1D/2D | Physics Constraint | Code |
|-------|------|--------|------------|-------|-------------------|------|
| Candes, Romberg, Tao | 2006 | L1 / Basis Pursuit | Theory | - | No | N/A |
| Candes & Fernandez-Granda | 2014 | TV norm on measures | Theory | - | No | N/A |
| Tang, Bhaskar, Recht | 2013 | Atomic Norm (ANM) | Theory | - | No | N/A |
| Bayindir | 2016 | CS + HOS spectral | Simulated | 1D | Nonlinear wave eq. | No |
| Bayindir | 2018 | L1 + directional basis | Simulated sparse spatial | 2D | Directional spectrum | No |
| Lyu et al. | 2025 | Nonneg. sparse reg. | HF radar Doppler | 1D | Radar backscatter model | No |
| Laface et al. | 2017/2022 | CS + harmonic wavelet | Pressure/time series | 1D | No | No |
| Harley & Moura | 2013 | L1 / Basis Pursuit | Sparse sensor array | 1D+disp. | **Dispersion relation** | **Yes** |
| Hu et al. | 2022 | Hybrid LASSO | Sparse sensors | 1D+disp. | Group-sparsity + continuity | No |
| Le Courtois & Bonnel | 2015 | CS + particle filter | 32-sensor HLA | 1D+disp. | **Dispersion tracking** | No |
| Desmars et al. | 2018/2023 | Spectral decomp. + HOS | Optical (dense) | 2D | Nonlinear wave eq. | No |
| Gallego/Yezzi et al. | 2021 | Sparse stereo + NUFFT | Stereo camera (sparse 3D) | 2D | **Linear dispersion** | Partial (WASS) |
| Wang & Pan | 2021/2022 | EnKF + HOS | Sparse probes | 2D | **Nonlinear wave eq.** | Partial |
| Feng Wang et al. | 2024 | ETKF | Optical 3D points | 2D | **Linear dispersion** | Partial |
| Ehlers et al. | 2023 | U-Net / FNO | Radar snapshots | 1D | Learned from HOS data | No |
| Ehlers et al. | 2025 | PINO | Buoy/radar (sparse) | 2D | **Free surface BCs** | No |
| SHRED | 2025 | LSTM + decoder | 1-2 buoys | 2D(SWH) | No | No |
| Bhaskar et al. | 2013 | Atomic Norm Denoising | Theory | 1D | No | CVX |
| Yang & Chu | 2019 | Decoupled ANM | Theory | 2D | No | No |

---

## 10. Relevance to WaveSplat

### Current WaveSplat Approach
- K=5-30 discrete spectral Gaussians with (kx, ky, omega, phi, amplitude)
- Beamforming initialization -> coarse-to-fine optimization
- Dispersion hard constraint: omega = sqrt(g|k|tanh(|k|d))
- 27-75 sparse LiDAR observation points
- nRMSE ~0.22 (linear), ~0.6-0.9 vs HOS ground truth

### Most Promising Directions for K=100+

#### Direction A: Grid-Based LASSO / Basis Pursuit
- **Idea**: Define a dense wavenumber grid (e.g., 50x50 = 2500 grid points in kx-ky space). Each grid point on the dispersion surface. Solve: min ||a||_1 s.t. ||y - Phi*a||_2 < epsilon, where Phi is the measurement matrix (observation points x grid points) and a is the sparse amplitude vector.
- **Pros**: Convex, single-shot, well-understood theory. Dispersion relation built into dictionary. Naturally handles K=100+ (most grid amplitudes will be zero).
- **Cons**: Grid mismatch (true wavenumbers may not align with grid). Dictionary coherence issues for nearby grid points.
- **Key references**: Harley & Moura (2013), Bayindir (2018), Lyu et al. (2025)

#### Direction B: Gridless / Atomic Norm Minimization
- **Idea**: Solve for continuous (kx, ky) without grid discretization using atomic norm minimization.
- **Pros**: No grid mismatch. Theoretical guarantees for super-resolution. Automatically determines K.
- **Cons**: 2D ANM requires SDP, computationally expensive. Decoupled ANM (Yang & Chu 2019) helps but still O(N^3). Adding dispersion constraint to ANM is non-trivial.
- **Key references**: Tang et al. (2013), Candes & Fernandez-Granda (2014), Yang & Chu (2019)

#### Direction C: Sparse Bayesian Learning
- **Idea**: Bayesian approach with sparsity-promoting prior on wavenumber grid amplitudes. Automatically determines K via relevance determination.
- **Pros**: Automatic model order selection. Handles uncertainty. More robust to dictionary coherence than L1.
- **Cons**: Iterative (slower). Grid-based unless extended. Complex implementation.
- **Key references**: Fang & Wang (2016 IEEE TSP), SBL literature

#### Direction D: Hybrid CS + Current WaveSplat
- **Idea**: Use LASSO/BP on a coarse grid for initialization (replacing beamforming), then refine with current optimization pipeline.
- **Pros**: Better initialization -> better final result. Leverages existing WaveSplat pipeline. CS handles the "how many components" question.
- **Cons**: Two-stage approach may not be globally optimal.
- **Key references**: Le Courtois & Bonnel (2015), Harley & Moura (2013)

### Key Insight from Literature

The closest analog to our problem is **Harley & Moura (2013)** from the SHM domain:
- Sparse sensors (like our LiDAR points)
- Dispersive waves (like ocean gravity waves)
- Wavenumber dictionary structured by dispersion relation
- L1 recovery with 10-20% spatial data

The main difference: their waves are in a bounded plate (discrete modes), while ocean waves have a continuous spectrum. This means we need either:
1. A sufficiently fine grid (Direction A) with appropriate regularization
2. A gridless method (Direction B) that handles the continuous nature
3. A Bayesian approach (Direction C) that marginalizes over model complexity

### Recommended Reading Priority (for implementation)

1. **Harley & Moura 2013** -- closest methodological analog, has code
2. **Bayindir 2016, 2018** -- direct ocean wave CS application
3. **Tang, Bhaskar, Recht 2013** -- gridless CS theory
4. **Le Courtois & Bonnel 2015** -- CS + dispersion tracking
5. **Gallego/Yezzi 2021** -- sparse points + dispersion for wave estimation (most similar input setup)

---

## References (Full List)

1. Candes, E., Romberg, J., Tao, T. (2006). "Robust Uncertainty Principles: Exact Signal Reconstruction from Highly Incomplete Frequency Information." IEEE Trans. Inf. Theory, 52(2).
2. Candes, E., Romberg, J., Tao, T. (2006). "Stable Signal Recovery from Incomplete and Inaccurate Measurements." Comm. Pure Appl. Math., 59(8), 1207-1223.
3. Candes, E., Fernandez-Granda, C. (2014). "Towards a Mathematical Theory of Super-Resolution." Comm. Pure Appl. Math., 67(6), 906-956.
4. Tang, G., Bhaskar, B., Recht, B. (2013). "Compressed Sensing Off the Grid." IEEE Trans. Inf. Theory, 59(11), 7465-7490.
5. Bhaskar, B., Tang, G., Recht, B. (2013). "Atomic Norm Denoising With Applications to Line Spectral Estimation." IEEE Trans. Inf. Theory, 59(6), 3958-3975.
6. Bayindir, C. (2016). "Compressive Spectral Method for the Simulation of the Nonlinear Gravity Waves." Scientific Reports, 6, 22100.
7. Bayindir, C. et al. (2018). "Extrapolation of Random Wave Field Data via Compressive Sampling." Ocean Engineering.
8. Lyu, et al. (2025). "Ocean Wave Spectrum Reconstruction from HF Radar Data and Its Application to Wave Height Estimation." arXiv:2505.13132.
9. Laface, V. et al. (2017). "Compressive Sensing-Based Reconstruction of Sea Free-Surface Displacement."
10. Laface, V. et al. (2022). "Compressive Sampling-Based Extrapolation of Free Surface Displacement Data from Pressure Measurements." Ocean Engineering.
11. Harley, J.B., Moura, J.M.F. (2013). "Sparse Recovery of the Multimodal and Dispersive Characteristics of Lamb Waves." J. Acoust. Soc. Am., 133(5), 2732-2745.
12. Hu, Y., Cui, F., Li, F., Tu, X., Zeng, L. (2022). "Sparse Wavenumber Analysis of Guided Wave Based on Hybrid Lasso Regression in Composite Laminates." Struct. Health Monit.
13. Le Courtois, F., Bonnel, J. (2015). "Compressed Sensing for Wideband Wavenumber Tracking in Dispersive Shallow Water." J. Acoust. Soc. Am., 138(2), 575-583.
14. Desmars, N. et al. (2018). "Phase-Resolved Reconstruction Algorithm and Deterministic Prediction of Nonlinear Ocean Waves From Spatio-Temporal Optical Measurements." ASME OMAE 2018.
15. Desmars, N., Hartmann, M., Behrendt, J., Hoffmann, N., Klein, M. (2023). "Nonlinear Deterministic Reconstruction and Prediction of Remotely Measured Ocean Surface Waves." J. Fluid Mech., 975.
16. Benetazzo, A., Bergamasco, F., Gallego, G., Yezzi, A. et al. (2021). "Toward Real-Time Optical Estimation of Ocean Waves' Space-Time Fields." Computers & Geosciences, 147, 104666.
17. Wang, G., Pan, Y. (2021). "Phase-Resolved Ocean Wave Forecast with Ensemble-Based Data Assimilation." J. Fluid Mech.
18. Wang, G., Pan, Y. (2022). "Phase-Resolved Ocean Wave Forecast with Simultaneous Current Estimation through Data Assimilation." J. Fluid Mech.
19. Wang, F., Zhu, Q., Cai, C., Wang, X., Qiao, R. (2024). "From Points to Waves: Fast Ocean Wave Spatial-Temporal Fields Estimation Using Ensemble Transform Kalman Filter with Optical Measurement." Coastal Engineering.
20. Ehlers, S., Klein, M. et al. (2023). "Machine Learning for Phase-Resolved Reconstruction of Nonlinear Ocean Wave Surface Elevations from Sparse Remote Sensing Data." Ocean Engineering, 288, 116059.
21. Ehlers, S. et al. (2025). "Bridging Ocean Wave Physics and Deep Learning: Physics-Informed Neural Operators for Nonlinear Wavefield Reconstruction in Real-Time." Physics of Fluids, 37(10), 107119.
22. Yang, Y., Chu, Z. (2019). "Efficient Two-Dimensional Line Spectrum Estimation Based on Decoupled Atomic Norm Minimization." Signal Processing.
23. Yang, Y. et al. (2018). "ADMM for ND Line Spectral Estimation using Grid-Free Compressive Sensing." arXiv:1811.02477.
24. Fang, J., Wang, F. et al. (2016). "Pattern-Coupled Sparse Bayesian Learning." IEEE Trans. Signal Processing, 64(18).
25. Bergamasco, F. et al. (2017). "WASS: An Open-Source Pipeline for 3D Stereo Reconstruction of Ocean Waves." Computers & Geosciences, 107, 28.
