"""Operator guidance for the parameter that SHAP names as the root cause."""

ACTIONS = {
    "frc": "Chlorine residual is off target. Check the biocide dosing system (risk of biofouling).",
    "do": "Dissolved oxygen is low. Possible microbial growth or overheating; check biocide and cooling fans.",
    "turb": "Turbidity is high. Suspended solids, biofilm or corrosion products; check filtration and inspect the loop.",
    "cond": "Conductivity is high. Cycles of concentration too high; check the blowdown valve.",
    "ph": "pH is out of range. Check acid/alkali dosing (scaling if high, corrosion if low).",
    "inhib": "Corrosion inhibitor is low. Check the inhibitor dosing pump.",
    "temp": "Water temperature is high. Check chillers, tower fans and the IT load.",
}
