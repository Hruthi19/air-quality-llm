"""AQI categorisation following the EPA breakpoint table for PM2.5.

The EPA PM2.5 (24-h average) breakpoint table is the canonical mapping from
raw PM2.5 concentration (ug/m^3) to the AQI 0-500 index. We reproduce it
here so downstream code (LLM grounding, hallucination detection) can ask
"is this AQI value consistent with this PM2.5 reading?" without going to
external resources.

References:
- US EPA. Technical Assistance Document for the Reporting of Daily Air Quality.
  EPA-454/B-18-007 (2018), Table 5.
"""
from __future__ import annotations

from dataclasses import dataclass

EPA_PM25_BREAKPOINTS = [
    (0.0,   12.0,    0,   50, "Good",
     "Air quality is satisfactory; little or no risk."),
    (12.1,  35.4,   51,  100, "Moderate",
     "Acceptable; unusually sensitive people should consider reducing prolonged exertion."),
    (35.5,  55.4,  101,  150, "Unhealthy for Sensitive Groups",
     "Sensitive groups (children, elderly, people with heart/lung disease) may experience effects."),
    (55.5, 150.4,  151,  200, "Unhealthy",
     "Everyone may begin to experience health effects; sensitive groups more serious effects."),
    (150.5, 250.4, 201,  300, "Very Unhealthy",
     "Health alert: everyone may experience more serious health effects."),
    (250.5, 500.4, 301,  500, "Hazardous",
     "Health warnings of emergency conditions; entire population more likely to be affected."),
]

CATEGORY_ACTION = {
    "Good":                          "Outdoor activity safe for everyone.",
    "Moderate":                      "Unusually sensitive people should consider reducing prolonged outdoor exertion.",
    "Unhealthy for Sensitive Groups":"Sensitive groups should reduce prolonged or heavy outdoor exertion.",
    "Unhealthy":                     "Everyone should reduce prolonged outdoor exertion; sensitive groups should avoid it.",
    "Very Unhealthy":                "Avoid prolonged outdoor exertion; sensitive groups should remain indoors.",
    "Hazardous":                     "Everyone should avoid all outdoor exertion; consider an N95 mask if outside.",
}


@dataclass
class AQIResult:
    pm25: float
    aqi: int
    category: str
    health_message: str
    recommended_action: str


def pm25_to_aqi(pm25: float) -> AQIResult:
    """Convert a PM2.5 concentration in ug/m^3 to an AQIResult (EPA piecewise linear)."""
    if pm25 is None or pm25 < 0:
        return AQIResult(pm25=float(pm25 or 0.0), aqi=0, category="Invalid",
                         health_message="Negative or missing PM2.5.",
                         recommended_action="No advice — input invalid.")
    pm25 = float(pm25)
    for c_lo, c_hi, i_lo, i_hi, name, msg in EPA_PM25_BREAKPOINTS:
        if pm25 <= c_hi:
            aqi = round((i_hi - i_lo) / (c_hi - c_lo) * (pm25 - c_lo) + i_lo)
            return AQIResult(pm25=pm25, aqi=int(aqi), category=name,
                             health_message=msg,
                             recommended_action=CATEGORY_ACTION[name])
    last = EPA_PM25_BREAKPOINTS[-1]
    return AQIResult(pm25=pm25, aqi=int(last[3]), category=last[4],
                     health_message=last[5],
                     recommended_action=CATEGORY_ACTION[last[4]])


def aqi_to_category_range(aqi: int) -> tuple[int, int, str]:
    """Inverse mapping: AQI int -> (lo, hi, category) for grounding checks."""
    for _, _, i_lo, i_hi, name, _ in EPA_PM25_BREAKPOINTS:
        if i_lo <= aqi <= i_hi:
            return i_lo, i_hi, name
    last = EPA_PM25_BREAKPOINTS[-1]
    return last[2], last[3], last[4]
