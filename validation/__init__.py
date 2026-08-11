"""C5.0 research validation apparatus. Offline only.

Import direction, from reports/c50/ARCHITECTURE.md §5.9: production never
imports research. Nothing under bot/ or signal_engine/ may import this
package. The dependency runs the other way — research reads what production
wrote (ledgers, frozen artifacts, config constants) and never the reverse.

M00 (cost_sensitivity) is the only module in this package so far, and the only
one in the whole C5.0 specification with no dependencies. It exists to answer
one question before any apparatus is built: is there a bar high enough that
clearing it is worth a quarter of work?
"""
