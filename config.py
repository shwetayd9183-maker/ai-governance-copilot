# config.py

CROP_ID_MAP = {
    "Onion": 23,
    "Tomato": 65,
    "Potato": 24
}

STATE_ID = 20  # Maharashtra

CROP_CONFIG = {
    "Onion": {"horizon": 7},
    "Tomato": {"horizon": 5},
    "Potato": {"horizon": 10}
}

SEVERITY_THRESHOLDS = {
    "moderate": 0.15,
    "severe": 0.25
}