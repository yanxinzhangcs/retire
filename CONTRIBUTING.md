# Contributing

Install the development dependencies and run the synthetic checks:

```bash
python -m pip install -e ".[dev]"
python -m pytest
python -m retire_serving.demo
python tools/check_public_tree.py
```

Use small synthetic examples to describe bugs. Keep experiment machine
identifiers, hardware inventories, cluster configuration, job identifiers,
account details, credentials, absolute local paths, raw traces, and execution
logs out of commits, issue bodies, and attachments.

Changes to the serving protocol should preserve current-version publication,
certified state inheritance, and retired-state isolation. Include a focused
test when changing those behaviors. Describe which path was exercised;
synthetic adapter tests do not establish model-serving performance.
