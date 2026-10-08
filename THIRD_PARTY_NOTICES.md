# License scope and third-party attribution

The root MIT license applies only to original contributions whose copyright is
held by the FedSPL authors. It does not override notices in individual files,
grant rights to third-party contributions, or relicense molecular datasets.
Mixed-origin files retain the rights of their respective contributors.

## FedChem

Repository: https://github.com/ur-whitelab/fedchem

Pinned official reference commit:
`55762d0c50c073e2367a6b5b90bca6967b21f612`.

FedSPL builds on FedChem's molecular federated-learning implementation. The
corrected snapshots are modified descendants, not verbatim upstream releases.
The `official_audit/` namespace preserves separately documented historical
reference experiments. No explicit top-level LICENSE file was found in the
pinned upstream listing checked on 8 October 2026. We do not assign MIT to
upstream-owned code or infer a license from public availability.

## MMGNN

Repository: https://github.com/MathIntelligence/MMGNN

Pinned upstream reference commit:
`78dcbff3a3d576506434241e4f26c70416453d9e`.

MMGNN is used in the separate encoder diagnostic. No explicit top-level LICENSE
file was found in this pinned upstream listing on 8 October 2026. Upstream
rights are retained; the FedSPL MIT grant does not extend to MMGNN-owned code.

## Other software and benchmark data

Existing in-file attribution and copyright notices are preserved, including
those in inherited FedML and molecular graph utilities. Dependencies such as
PyTorch, DGL, DGLLife, RDKit, NumPy and scikit-learn remain subject to their
own licenses and are not redistributed as installed environments here.

BBBP, SIDER, Tox21, HIV, ToxCast, ESOL, FreeSolv and Lipophilicity are reused
public benchmark inputs. They are not newly collected data and are not covered
by the code MIT license. Source-declared acquisition locations and preprocessing
details are recorded in [ENVIRONMENT.md](reproducibility/ENVIRONMENT.md).
Users must observe the original data providers' terms and acknowledge the
underlying datasets as well as the distribution tools.

This notice records provenance and the limits of our license grant; it is not
a representation that all third-party material has the same open-source license.
