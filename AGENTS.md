Per-act, per-edge intervention framework for starsim/stisim sexual networks. Python 3.10+.

## Commands
- **Test**: `SCIRIS_BACKEND=agg pytest tests/`
- **Test (single)**: `SCIRIS_BACKEND=agg pytest tests/test_starsim_only.py::test_name -x`

## Testing
pytest with plain `assert`. Each test file has a `make_sim()` factory and an `if __name__ == '__main__'` block for standalone execution. Tests return objects (warning suppressed in pytest.ini). Decorate with `@sc.timer()`.

```python
@sc.timer()
def test_my_feature():
    sim = make_sim(my_flag=True)
    sim.run()
    assert sim.results.sir.n_infected[-1] < 100, f"Expected <100, got {sim.results.sir.n_infected[-1]}"
    return sim
```

## Architecture pointers
- For architecture overview and code navigation: `README.md`
- For the starsim-only layer (no stisim dependency): `enroute/core.py`
- For stisim structured sexual networks: `enroute/sti.py`
- For a complete intervention example: `enroute/doxypep/`
- For example custom interventions: `enroute_examples/`

## Git workflow
Conventional commits (`feat:`, `fix:`, `refactor:`, `test:`, `docs:`).

## Boundaries
- **Always**: run `SCIRIS_BACKEND=agg pytest tests/` before committing
- **Ask first**: new runtime dependencies (anything added to `pyproject.toml [project].dependencies`)
- **Never** import `stisim` in `enroute/core.py` → use `enroute/sti.py` for stisim-dependent code

## Gotchas
- `SCIRIS_BACKEND=agg` is required for all test/example runs — without it, matplotlib fails on headless systems.
- `enroute_examples/` is a distributed package (not just scripts) — its classes are imported by tests.
- `EdgeIntervention` subclasses auto-register a `<name>_uses` column on the network edge table; use this instead of manually creating edge columns for intervention tracking.
