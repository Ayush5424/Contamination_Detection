"""Physics-informed cooling-loop simulator - paper Section IV-A.

Generates 5-minute sensor streams for an open recirculating cooling tower, injects
contamination events of five types with ground-truth labels, and adds measurement
noise, sensor faults and missing readings.
"""
import numpy as np
import pandas as pd

from .cwqi import SENSORS, cwqi

DT_MIN = 5                                   # sampling interval
STEPS_PER_HOUR = 60 // DT_MIN

NOISE = dict(temp=0.15, ph=0.05, cond=25.0, do=0.15, frc=0.03, turb=0.25, inhib=0.3)

EVENT_TYPES = ["biofouling", "scaling", "corrosion", "inhibitor_loss", "thermal"]
EVENT_LABEL = {"biofouling": "Biofouling (biocide depleted)",
               "scaling": "Scaling (stuck blowdown valve)",
               "corrosion": "Corrosion (acid overfeed)",
               "inhibitor_loss": "Inhibitor dosing failure",
               "thermal": "Thermal excursion"}
# parameters the fault physically acts on (used to score root-cause identification)
EVENT_ROOT = {"biofouling": {"frc", "do", "turb"},
              "scaling": {"cond", "ph"},
              "corrosion": {"ph", "turb", "inhib"},
              "inhibitor_loss": {"inhib"},
              "thermal": {"temp", "do"}}


def do_saturation(temp_c):
    """Standard empirical dissolved-oxygen saturation curve (mg/L)."""
    return 14.62 - 0.3898 * temp_c + 0.006969 * temp_c ** 2 - 0.00005897 * temp_c ** 3


def _ar1(n, sigma, phi, rng):
    """First-order autoregressive noise, so the drift looks like a real process."""
    e = rng.normal(0, sigma, n)
    x = np.empty(n)
    x[0] = e[0]
    for i in range(1, n):
        x[i] = phi * x[i - 1] + e[i]
    return x


def _envelope(n_steps, rng):
    """Gradual onset (35-50% of the event), plateau, then recovery = corrective action."""
    ramp = int(n_steps * rng.uniform(0.35, 0.5))
    recovery = int(n_steps * 0.2)
    plateau = n_steps - ramp - recovery
    up = 0.5 - 0.5 * np.cos(np.linspace(0, np.pi, ramp))
    down = 0.5 + 0.5 * np.cos(np.linspace(0, np.pi, recovery))
    return np.concatenate([up, np.ones(plateau), down])


def apply_event(readings, event_type, severity):
    """Apply one contamination event to a single set of true readings.

    Used both by the simulator and by the dashboard's "inject fault" buttons, so the
    demo feeds the models physically consistent values instead of random numbers.
    """
    r = dict(readings)
    s = float(severity)
    if event_type == "biofouling":
        r["frc"] = r["frc"] * max(0.0, 1 - 0.9 * s)
        r["do"] = max(0.0, r["do"] - 3.0 * s)
        r["turb"] = r["turb"] + 11.0 * s
    elif event_type == "scaling":
        r["cond"] = r["cond"] + 900.0 * s
        r["ph"] = r["ph"] + 0.55 * s
        r["turb"] = r["turb"] + 3.0 * s
    elif event_type == "corrosion":
        r["ph"] = r["ph"] - 1.3 * s
        r["inhib"] = max(0.0, r["inhib"] - 3.0 * s)
        r["turb"] = r["turb"] + 6.0 * s
    elif event_type == "inhibitor_loss":
        r["inhib"] = max(0.0, r["inhib"] - 6.5 * s)
        r["turb"] = r["turb"] + 1.5 * s
    elif event_type == "thermal":
        r["temp"] = r["temp"] + 6.5 * s
        r["do"] = max(0.0, r["do"] - 0.8 * s)
        r["frc"] = r["frc"] * max(0.0, 1 - 0.35 * s)
    else:
        raise ValueError(f"unknown event type: {event_type}")
    return r


def simulate(days=240, seed=42):
    """Return (true_values, measured_values, labels, events, faults)."""
    rng = np.random.default_rng(seed)
    n = days * 24 * STEPS_PER_HOUR
    hours = np.arange(n) * DT_MIN / 60.0

    # IT load: daily and weekly cycles plus slow random variation
    weekday = ((hours // 24) % 7 < 5).astype(float)
    load = 0.72 + 0.12 * np.sin(2 * np.pi * (hours - 9) / 24) + 0.05 * weekday + _ar1(n, 0.01, 0.98, rng)
    load = np.clip(load, 0.4, 1.0)
    ambient = 27 + 5 * np.sin(np.pi * hours / (days * 24)) + 3.5 * np.sin(2 * np.pi * (hours - 15) / 24)

    # ---- contamination events (ground truth) ----
    severity = {k: np.zeros(n) for k in EVENT_TYPES}
    events = []
    i = int(rng.uniform(24, 48) * STEPS_PER_HOUR)
    while True:
        duration = int(rng.uniform(8, 30) * STEPS_PER_HOUR)
        if i + duration >= n - STEPS_PER_HOUR:
            break
        etype = EVENT_TYPES[len(events) % len(EVENT_TYPES)] if len(events) < 5 else str(rng.choice(EVENT_TYPES))
        magnitude = rng.uniform(0.8, 2.0)
        severity[etype][i:i + duration] = np.maximum(severity[etype][i:i + duration],
                                                     _envelope(duration, rng) * magnitude)
        events.append(dict(type=etype, start=i, end=i + duration, magnitude=float(magnitude)))
        i += duration + int(max(12, rng.exponential(55)) * STEPS_PER_HOUR)

    bio, scale, corr, inhib_loss, thermal = (severity[k] for k in EVENT_TYPES)

    # ---- true process values ----
    temp = 24 + 6.5 * load + 0.25 * (ambient - 27) + _ar1(n, 0.08, 0.95, rng) + 6.5 * thermal

    cond = np.empty(n)                       # evaporative concentration + blowdown controller
    c, blowing = 1900.0, False
    for k in range(n):
        c += 3.2 * load[k] + rng.normal(0, 1.0)
        if scale[k] > 0.05:                  # valve stuck: no blowdown, salts keep rising
            blowing = False
            c += 6.0 * scale[k]
        else:
            if c > 2400:
                blowing = True
            if blowing:
                c -= 35.0
                if c < 1850:
                    blowing = False
        cond[k] = c = max(c, 1500 + 400 * (scale[k] > 0.05))

    ph = (8.05 + 0.25 * (cond - 2000) / 1000 + 0.03 * np.sin(2 * np.pi * hours / 24)
          + _ar1(n, 0.01, 0.97, rng) + 0.55 * scale - 1.3 * corr)

    hours_since_dose = hours % 8.0           # halogen slug every 8 h, temperature-dependent decay
    tau = 6.5 / (1 + 0.04 * (temp - 29))
    frc = 0.85 * np.exp(-hours_since_dose / tau) + 0.08
    frc = frc * (1 - 0.9 * np.clip(bio, 0, 1)) * (1 - 0.35 * np.clip(thermal, 0, 1))
    frc = np.clip(frc + _ar1(n, 0.005, 0.9, rng), 0.0, None)

    do = 0.93 * do_saturation(temp) + _ar1(n, 0.03, 0.95, rng) - 3.0 * bio - 0.8 * thermal
    turb = (1.4 + 0.4 * load + np.abs(_ar1(n, 0.05, 0.9, rng))
            + 11 * bio + 3 * scale + 6 * corr + 1.5 * inhib_loss)
    inhib = 10.0 + _ar1(n, 0.05, 0.97, rng) - 3.0 * corr - 6.5 * inhib_loss

    true = pd.DataFrame({"temp": temp, "ph": ph, "cond": cond, "do": np.clip(do, 0, None),
                         "frc": frc, "turb": np.clip(turb, 0, None), "inhib": np.clip(inhib, 0, None)},
                        index=pd.date_range("2026-01-01", periods=n, freq=f"{DT_MIN}min"))

    # ---- measurements: noise, sensor faults, missing samples ----
    measured = true.copy()
    for s in SENSORS:
        measured[s] = true[s] + rng.normal(0, NOISE[s], n)
    fault = np.zeros(n, dtype=bool)
    faults = []
    for _ in range(40):                                        # short spikes
        k, s, length = rng.integers(0, n - 3), str(rng.choice(SENSORS)), int(rng.integers(1, 4))
        measured.iloc[k:k + length, SENSORS.index(s)] += rng.choice([-1, 1]) * rng.uniform(8, 15) * NOISE[s] * 4
        fault[k:k + length] = True
        faults.append(dict(kind="spike", sensor=s, start=int(k), end=int(k + length)))
    for _ in range(10):                                        # stuck (flat-line) probe
        k, s = rng.integers(0, n - 40), str(rng.choice(SENSORS))
        length = int(rng.uniform(1, 3) * STEPS_PER_HOUR)
        measured.iloc[k:k + length, SENSORS.index(s)] = measured.iloc[k, SENSORS.index(s)]
        fault[k:k + length] = True
        faults.append(dict(kind="flatline", sensor=s, start=int(k), end=int(k + length)))
    measured = measured.mask(rng.random((n, len(SENSORS))) < 0.01)     # 1% missing

    event_type = np.full(n, "none", dtype=object)
    for e in events:
        event_type[e["start"]:e["end"]] = e["type"]
    labels = pd.DataFrame({"load": load,
                           "cwqi_true": cwqi(true),
                           "event": np.vstack([severity[k] for k in EVENT_TYPES]).max(0) > 0.05,
                           "event_type": event_type,
                           "fault": fault}, index=true.index)
    return true, measured, labels, events, faults
