PYTHON ?= python
RUNTIME ?= cpu

.PHONY: verify verify-declared verify-inventory
verify:
	$(PYTHON) scripts/verify_declared_v1.py --runtime $(RUNTIME) --require-complete

verify-declared:
	$(PYTHON) scripts/verify_declared_v1.py --runtime $(RUNTIME)

verify-inventory:
	$(PYTHON) scripts/verification_inventory_v1.py
