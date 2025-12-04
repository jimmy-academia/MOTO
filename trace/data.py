TRIAGE_SPEC = "If (O₂ saturation < 92 OR systolic BP < 90 OR HR > 130):\n    → severity = critical\nElse If (O₂ < 95 OR Temp > 101°F OR RR > 24):\n    → severity = moderate\nElse:\n    → severity = mild\n\nIf (≥2 chronic conditions OR age ≥ 70):\n    → risk = high\nElse:\n    → risk = standard\n\nIf (severity = critical):\n    → recommend ER referral\nElse If (severity = moderate AND risk = high):\n    → recommend urgent clinical evaluation\nElse If (severity = moderate AND risk = standard):\n    → recommend outpatient evaluation\nElse:\n    → recommend home care"

HEALTHCARE_EXAMPLES = [
    {
        "query": "A 51-year-old patient presents with loss of smell for 1 days, chest pain for 7 days, headache for 7 days. Oxygen saturation is 90%, and body temperature is 102.0°F. Medical history includes HIV. Recent travel to outbreak area.",
        "label": "ER referral"
    },
    {
        "query":"A 89-year-old patient presents with headache for 8 days, rash for 6 days, chest pain for 4 days, dry cough for 8 days, shortness of breath for 3 days. Oxygen saturation is 94%, and body temperature is 98.6°F. Medical history includes chronic kidney disease. Recent travel abroad.",
        "label":"Urgent clinical evaluation"
    },
    {
        "query":"A 16-year-old patient presents with fever for 8 days, shortness of breath for 4 days. Oxygen saturation is 90%, and body temperature is 103.0°F. Medical history includes CHF, hypertension. Recent travel abroad.",
        "label":"ER referral"
    },
    {
        "query":"A 37-year-old patient presents with loss of smell for 7 days, fever for 9 days. Oxygen saturation is 97%, and body temperature is 99.5°F. Medical history includes none. Recent travel abroad.",
        "label":"Home care"
    },
    {
        "query":"A 59-year-old patient presents with chest pain for 10 days, nausea for 7 days, fatigue for 4 days. Oxygen saturation is 97%, and body temperature is 99.5°F. Medical history includes diabetes, hypertension. Recent travel abroad.",
        "label":"Home care"
    },
]