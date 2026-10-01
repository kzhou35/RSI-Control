"""RSI Control Arena: can an AI R&D agent covertly sabotage the successor model it trains?

``tasks`` pins each (main task, side task) pair, ``goals`` defines the side tasks, ``evals`` holds
the held-out oracles and cover metrics, ``clients`` the model clients they share. The agent pipeline
and scorers are in ``scripts/``; see the top-level README.
"""

__version__ = "1.0.0"
