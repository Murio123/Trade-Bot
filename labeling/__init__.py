"""C5.0 labeling primitives. Offline research only.

M02 (`triple_barrier`) turns a decision bar into a path-dependent outcome; M03
(`sample_weights`) turns those outcomes' spans into honest sample sizes. Both
are pure: no I/O, no exchange access, no database, no config reads.

Import direction, from reports/c50/ARCHITECTURE.md §5.9: production never
imports research. Nothing under bot/ or signal_engine/ may import this package,
and nothing in this package may import the `tools.*` offline layer — the
dependency runs tools -> labeling, never back.

Why a package of its own rather than another module under validation/: the
paths are fixed by ARCHITECTURE.md §1, and the split is meaningful. `validation`
decides whether a measurement is trustworthy; `labeling` decides what was
measured.
"""
