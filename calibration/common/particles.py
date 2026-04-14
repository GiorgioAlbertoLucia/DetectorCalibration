"""
Central particle registry.

Keeps PDG codes, mass aliases, and the ITS/TPC suffix convention
in one place so every submodule imports from here instead of
redefining its own dicts.
"""

from particle import Particle

# ── PDG codes ─────────────────────────────────────────────────────────────────
PDG_CODE: dict[str, int] = {
    'Pi': 211,
    'Ka': 321,
    'Pr': 2212,
    'De': 1000010020,
    'He': 1000020030,   # He-3
}

# ── Masses in GeV/c² (derived from PDG, matching torchic convention) ──────────
PARTICLE_MASS: dict[str, float] = {
    name: Particle.from_pdgid(pdg).mass / 1_000
    for name, pdg in PDG_CODE.items()
}
# Aliases used elsewhere in the codebase
PARTICLE_MASS['He3'] = PARTICLE_MASS['He']

# ── Tree/column suffix used in the O2 table ───────────────────────────────────
# e.g. fPtHad, fItsClusterSizeHe3
TREE_SUFFIX: dict[str, str] = {
    'Pr': 'Had',
    'He': 'He3',
    'He3': 'He3',
    'Had': 'Had',
}