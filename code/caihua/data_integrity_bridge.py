"""Locate the shared clean-data protocol relative to this project."""
from pathlib import Path
import importlib.util, sys
_here=Path(__file__).resolve()
_path=next((p/'data_integrity/protocol.py' for p in _here.parents if (p/'data_integrity/protocol.py').is_file()),None)
if _path is None: raise RuntimeError("Missing data_integrity/protocol.py; do not use legacy splits")
_name="pidnet_clean_data_protocol"
if _name not in sys.modules:
    _spec=importlib.util.spec_from_file_location(_name,str(_path))
    _module=importlib.util.module_from_spec(_spec);sys.modules[_name]=_module;_spec.loader.exec_module(_module)
_module=sys.modules[_name]
open_split=_module.open_split
training_preflight=_module.training_preflight
write_run_stamp=_module.write_run_stamp
validate_resume=_module.validate_resume
validate_evaluation=_module.validate_evaluation
