"""Never-ending live version of the cooling-loop simulator, one 5-min reading at a time.

simulator.simulate() builds a whole data set in one go (used for training and testing).
PlantStream uses the same process equations, noise levels and event physics, but advances
step by step, so a sensor feeder can publish fresh readings forever without any CSV file.

    stream = PlantStream(seed=None)          # random seed: a different plant run every time
    ts, measured, true_cwqi, info = stream.next()

Natural contamination events start on their own now and then (same five types, envelope and
magnitudes as the simulator, but closer together so a demo sees one within minutes); an
operator can also start one with inject() and stop it with clear().
"""
import numpy as np
import pandas as pd

from .cwqi import SENSORS, cwqi
from .simulator import DT_MIN, EVENT_TYPES, NOISE, STEPS_PER_HOUR, do_saturation


class _AR1:
    """First-order autoregressive noise, started in its stationary state."""

    def __init__(self, sigma, phi, rng):
        self.sigma, self.phi, self.rng = sigma, phi, rng
        self.x = rng.normal(0, sigma / np.sqrt(1 - phi ** 2))

    def __call__(self):
        self.x = self.phi * self.x + self.rng.normal(0, self.sigma)
        return self.x


class PlantStream:
    def __init__(self, seed=None, start=None, auto_events=True, first_event_hours=(3, 6),
                 gap_hours=(12, 30), sensor_faults=True):
        self.rng = rng = np.random.default_rng(seed)
        self.t = pd.Timestamp(start if start is not None else pd.Timestamp.now()).floor(f"{DT_MIN}min")
        self.auto_events, self.gap_hours, self.sensor_faults = auto_events, gap_hours, sensor_faults
        self.ar = dict(load=_AR1(0.01, 0.98, rng), temp=_AR1(0.08, 0.95, rng), ph=_AR1(0.01, 0.97, rng),
                       frc=_AR1(0.005, 0.9, rng), do=_AR1(0.03, 0.95, rng), turb=_AR1(0.05, 0.9, rng),
                       inhib=_AR1(0.05, 0.97, rng))
        self.cond, self.blowing = 1900.0 + rng.uniform(-50, 300), False
        self.event = None                         # natural event: type, profile, k
        self.manual = None                        # operator injection: type, severity, ramp, k, clearing
        self.next_event = int(rng.uniform(*first_event_hours) * STEPS_PER_HOUR)
        self.stuck = None                         # flat-lined probe: sensor, value, steps left
        self.spike = None                         # short spike: sensor, offset, steps left

    # ------------------------------------------------------------ operator commands
    def inject(self, event_type, severity=1.5, ramp_hours=3.0):
        if event_type not in EVENT_TYPES:
            raise ValueError(f"event type must be one of {EVENT_TYPES}")
        if self.event:                            # one fault at a time: a running natural event fades out
            self._fade_natural()
        self.manual = dict(type=event_type, severity=float(severity),
                           ramp=max(1, int(float(ramp_hours) * STEPS_PER_HOUR)), k=0, clearing=None)

    def clear(self):
        """Corrective action: an injected or natural event fades out over one hour."""
        if self.manual and self.manual["clearing"] is None:
            self.manual["clearing"] = 0
        if self.event:
            self._fade_natural()

    def _fade_natural(self):
        k = self.event["k"]
        prof = self.event["profile"]
        level = prof[min(k, len(prof) - 1)]
        self.event["profile"] = np.concatenate([prof[:k], level * np.linspace(1, 0, STEPS_PER_HOUR)])

    def active_event(self):
        """What is going on in the water right now (for the status message)."""
        out = []
        if self.event:
            out.append({"type": self.event["type"], "source": "natural", "magnitude": self.event["magnitude"]})
        if self.manual:
            out.append({"type": self.manual["type"], "source": "injected", "severity": self.manual["severity"],
                        "clearing": self.manual["clearing"] is not None})
        return out

    # ------------------------------------------------------------ one step
    def _severity(self):
        sev = dict.fromkeys(EVENT_TYPES, 0.0)
        if self.event is None and self.auto_events and self.manual is None:
            self.next_event -= 1
            if self.next_event <= 0:              # same duration, envelope and magnitude as simulate()
                n = int(self.rng.uniform(8, 30) * STEPS_PER_HOUR)
                ramp = int(n * self.rng.uniform(0.35, 0.5))
                recovery = int(n * 0.2)
                prof = np.concatenate([0.5 - 0.5 * np.cos(np.linspace(0, np.pi, ramp)),
                                       np.ones(n - ramp - recovery),
                                       0.5 + 0.5 * np.cos(np.linspace(0, np.pi, recovery))])
                magnitude = float(self.rng.uniform(0.8, 2.0))
                self.event = dict(type=str(self.rng.choice(EVENT_TYPES)), profile=prof * magnitude,
                                  magnitude=magnitude, k=0)
        if self.event:
            e = self.event
            sev[e["type"]] = max(sev[e["type"]], float(e["profile"][e["k"]]))
            e["k"] += 1
            if e["k"] >= len(e["profile"]):
                self.event = None
                self.next_event = int(self.rng.uniform(*self.gap_hours) * STEPS_PER_HOUR)
        if self.manual:
            m = self.manual
            s = m["severity"] * min(1.0, (m["k"] + 1) / m["ramp"])
            if m["clearing"] is not None:
                s *= max(0.0, 1 - (m["clearing"] + 1) / STEPS_PER_HOUR)
                m["clearing"] += 1
            sev[m["type"]] = max(sev[m["type"]], s)
            m["k"] += 1
            if m["clearing"] is not None and m["clearing"] >= STEPS_PER_HOUR:
                self.manual = None
        return sev

    def next(self):
        """Returns (time, measured readings with NaN for missing, ground-truth CWQI, active events)."""
        rng, ar, t = self.rng, self.ar, self.t
        hod = t.hour + t.minute / 60.0
        sev = self._severity()
        bio, scale, corr, inhib_loss, thermal = (sev[k] for k in EVENT_TYPES)

        weekday = float(t.dayofweek < 5)
        load = float(np.clip(0.72 + 0.12 * np.sin(2 * np.pi * (hod - 9) / 24) + 0.05 * weekday + ar["load"](),
                             0.4, 1.0))
        season = np.sin(np.pi * ((t.dayofyear - 1) / 365.0))
        ambient = 27 + 5 * season + 3.5 * np.sin(2 * np.pi * (hod - 15) / 24)

        temp = 24 + 6.5 * load + 0.25 * (ambient - 27) + ar["temp"]() + 6.5 * thermal
        c = self.cond + 3.2 * load + rng.normal(0, 1.0)
        if scale > 0.05:                          # blowdown valve stuck: salts keep rising
            self.blowing = False
            c += 6.0 * scale
        else:
            if c > 2400:
                self.blowing = True
            if self.blowing:
                c -= 35.0
                if c < 1850:
                    self.blowing = False
        self.cond = c = max(c, 1500 + 400 * (scale > 0.05))
        ph = 8.05 + 0.25 * (c - 2000) / 1000 + 0.03 * np.sin(2 * np.pi * hod / 24) + ar["ph"]() \
            + 0.55 * scale - 1.3 * corr
        tau = 6.5 / (1 + 0.04 * (temp - 29))
        frc = (0.85 * np.exp(-(hod % 8.0) / tau) + 0.08) * (1 - 0.9 * min(bio, 1)) * (1 - 0.35 * min(thermal, 1))
        frc = max(frc + ar["frc"](), 0.0)
        do = max(0.93 * do_saturation(temp) + ar["do"]() - 3.0 * bio - 0.8 * thermal, 0.0)
        turb = max(1.4 + 0.4 * load + abs(ar["turb"]()) + 11 * bio + 3 * scale + 6 * corr + 1.5 * inhib_loss, 0.0)
        inhib = max(10.0 + ar["inhib"]() - 3.0 * corr - 6.5 * inhib_loss, 0.0)

        true = dict(temp=temp, ph=ph, cond=c, do=do, frc=frc, turb=turb, inhib=inhib)
        true_cwqi = float(cwqi(pd.DataFrame([true]))[0])
        measured = {s: true[s] + rng.normal(0, NOISE[s]) for s in SENSORS}

        if self.sensor_faults:                    # same kinds of probe faults as the simulator, at its rates
            if self.spike is None and rng.random() < 40 / (240 * 288):
                s = str(rng.choice(SENSORS))
                self.spike = dict(sensor=s, steps=int(rng.integers(1, 4)),
                                  offset=rng.choice([-1, 1]) * rng.uniform(8, 15) * NOISE[s] * 4)
            if self.spike:
                measured[self.spike["sensor"]] += self.spike["offset"]
                self.spike["steps"] -= 1
                if self.spike["steps"] <= 0:
                    self.spike = None
            if self.stuck is None and rng.random() < 10 / (240 * 288):
                s = str(rng.choice(SENSORS))
                self.stuck = dict(sensor=s, value=measured[s], steps=int(rng.uniform(1, 3) * STEPS_PER_HOUR))
            if self.stuck:
                measured[self.stuck["sensor"]] = self.stuck["value"]
                self.stuck["steps"] -= 1
                if self.stuck["steps"] <= 0:
                    self.stuck = None
            for s in SENSORS:                     # 1% missing samples
                if rng.random() < 0.01:
                    measured[s] = np.nan

        self.t = t + pd.Timedelta(minutes=DT_MIN)
        return t, measured, true_cwqi, self.active_event()
