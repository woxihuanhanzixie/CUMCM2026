"""Export is implemented in export_results.py; never call the received writer."""
def write_result2(*a, **k):
    raise RuntimeError("Use export_results.py with verified E0/E144")

def check_result2(*a, **k):
    raise RuntimeError("Use audit_workbook.py independently")
