"""Shared building blocks for the PO4AO variants in clauding/ (pix, pol, pinn, reward).

    toy_env.py    fast frozen-flow AO simulator (torch) with the same step()/flatten() interface as the bench
    bench_dao.py  the same interface on the PAPYRUS DAO shared memories (not testable off the bench)
    replay.py     ring buffer that only returns windows that are contiguous in time
    models.py     the baseline PO4AO networks (port of po4ao_models_PAPYRUS_upd.py)
    core.py       the PO4AO agent: dynamics training, policy training, history handling
    runner.py     synchronous loop for simulation, asynchronous (trainer process) loop for the bench
"""
