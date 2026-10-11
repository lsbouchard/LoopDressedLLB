# Numerical Audit, October 2026

## Status

The software errors listed below are corrected and covered by regression
tests. The published separable `N=0` null control remains unreproduced by the
implemented covariance equations. No response-dependent cutoff, forced zero,
or fitted subtraction has been introduced to obtain that result.

The baseline for this audit is commit
`7162ecc1e5775c034e2b8b848dac210d9372585c`. Earlier results remain available in
Git history and in the legacy notebook; they are not current validation data.

## Response Invariant And Covariance Null Control

At `N=0`, the deterministic response derivative vanishes. This property does
not make the magnetization fluctuation generator vanish. The dipolar derivative
`delta M_j x B_j + M_j x k sum_g T_jg delta M_g` still couples packets.
An independently assembled analytic Jacobian agrees with the finite-difference
generator. Its symmetric part is nonzero, while its Hamiltonian condition
`A Omega + Omega A^T = 0` holds to finite-difference accuracy in the benchmark.
Thus zero diffusion does not imply a stationary vacuum covariance.

At `k=0.03`, `t=8`, tip `0.05`, zero response and zero loops, direct propagation
`V(t) = exp(A t) V(0) exp(A^T t)` gives a maximum nearest-neighbor logarithmic
negativity of approximately `0.7111304971`. The result persists for response
amplitude `1e-8`, and the covariance satisfies the uncertainty bound to numerical
precision. Zero coupling, in contrast, gives zero entanglement.

The earlier Python CLI bypassed covariance propagation at zero response and
returned predetermined separable metrics. The C program propagated the
covariance. Agreement in four nonzero-response endpoint tests did not test this
difference. Python now propagates the same covariance at zero response; both
programs identify the disagreement explicitly. `--ent-native-model none` means
no computed result, not zero entanglement.

Resolving the scientific discrepancy requires reconciling the intended
quadratic fluctuation generator and subsystem definition with the null-control
statement. This audit does not establish that the published theory is wrong,
or that a modified generator reproduces it.

## Corrected Errors

1. **Non-tangent covariance injection.** The old perturbation changed `M` while
   keeping `N` fixed, then normalized both branch vectors. For transverse `N`,
   the derivative of that normalization contracts an `M` displacement along
   `N` even at zero elapsed time. The corrected lift rotates `N` with `M`:
   `delta N = -M (N . delta M) / |M|^2`. This satisfies the linearized norm and
   orthogonality constraints and leaves the canonical displacement unchanged
   under constraint restoration to finite-difference accuracy. It corresponds
   to a common infinitesimal rotation of the packet background, with no
   additional rotation around `M`. Both frozen and trajectory models use it.
2. **Incomplete bipartitions.** Python simultaneously required an anchor on the
   left and discarded larger left sets. That omitted distinct partitions.
   C also tested only the first packet against the other two. All `2^(m-1)-1`
   distinct partitions are now enumerated. Full inseparability uses the weakest
   partition within each subset. A squeezed pair plus a vacuum third packet
   correctly has zero full-inseparability score under every relabeling.
3. **Insufficient physicality checks.** Absolute eigenvalues of `i Omega V` can
   incorrectly accept an indefinite matrix. Physicality is now checked with the
   minimum Hermitian eigenvalue of `V + i Omega/2`. Isotropic repair is the
   smallest shift restoring this matrix's positivity, with a `1e-9` safety
   margin; violations within `1e-10` are treated as numerical roundoff.
   Symplectic spectra use a positive Cholesky factor and a Hermitian eigensolve.
4. **Ignored mean aggregation.** C parsed `--ent-aggregate mean` but computed
   maxima. It now implements the requested aggregate for all reported metrics.
5. **Repeated sample indices.** Duplicate indices overwrote Python lookup
   entries and stalled C sampling. Every requested bin is now populated.
6. **Unsupported and malformed C options.** Misspelled flags, unsupported model
   selections, nonfinite values, invalid windows and zero-magnetization initial
   states now fail explicitly. Numerical eigensolver errors propagate to a
   failed run, and pthread creation failure falls back to available workers.
   Static mixing also follows Python's field cap without the dynamic-only
   amplitude cap.
7. **Misleading dressing diagnostics.** The reported back-action scale used a
   different power of response amplitude from the actual edge kernel. It now
   reports the applied edge factors. The unused vertex multiplier is reported
   as unity. Roundoff-level PPT and Duan threshold crossings are no longer
   counted as an entangled fraction of the observation window.

## Validity Diagnostics

Canonical basis reconstruction is separate from quantum-channel validity.
For a frozen generator, the additional check is
`D - i (A Omega + Omega A^T)/2 >= 0`. For a discrete map it is
`Q + i (Omega - Phi Omega Phi^T)/2 >= 0`. These are the Gaussian complete-
positivity conditions, in the convention where vacuum covariance is `I/2`.
See [Weedbrook et al., Gaussian Quantum Information, Secs. II and V](https://arxiv.org/abs/1110.3234).

The frozen benchmark's coherent generator passes this check within the `1e-8`
finite-difference tolerance. General moving-background and dissipative cases
require their own checks. In the trajectory model, the discrete map is tested;
the local frozen generator diagnostic lacks the moving-frame connection and
does not by itself certify or reject the trajectory map. Covariance repair
cannot establish that an underlying evolution is a quantum channel.

C reports `generator_cp_margin`, the minimum pre-repair uncertainty margin,
maximum added noise, accepted/sample counts, the response invariant, and any
null-control violation. Failed generator checks or repairs exceeding
`--ent-native-reg-warn` exclude samples from summaries. If no samples remain,
JSON metrics are `null`. Endpoint fields refer to the last accepted sample;
the accepted count must be checked. Python exports corresponding diagnostics
and excludes failed basis, channel or uncertainty checks.

## Performance And Verification

Frozen evolution uses exact Gaussian steps between requested samples, rather
than one multiplication per integration-grid point. The grid still defines
the requested sample times. C reuses its generator, uses contiguous matrix
multiplication and bounded eigensolver workspaces, and retains parallelism
over coupling values. Metal is an optional shader compilation probe, not an
accelerated numerical backend.

On the local Apple Silicon machine, a seven-run timing check of one frozen
covariance trajectory (17 samples on a 2401-point grid, excluding warm-up)
gave median times of about 24 ms before the audit and 11 ms after it. This
is a small-kernel measurement, not an end-to-end or GPU speed claim.

Tests cover analytic vacuum, thermal, squeezed and biseparable states;
independent analytic null-branch Jacobians; constrained tangent identities;
all bipartitions through eight modes; thermal attenuation; repeated and invalid
sample times; thread determinism; and full-precision C/Python comparisons across
couplings and summaries. The comparison tolerance is `2e-8`, replacing the old
`5e-3` check against rounded CLI endpoints.

This audit concentrates on packet covariance, Gaussian diagnostics and the C
port. It does not validate all exploratory stochastic, memory, plotting or
engineered-control workflows, or establish a microscopic derivation of the
phenomenological closure. The original notebook is retained unchanged.
