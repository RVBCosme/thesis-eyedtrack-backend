# alert_levels.py
from enum import Enum

class AlertLevel(Enum):
    INFO = "info"
    WARNING = "warning"
    DANGER = "danger"