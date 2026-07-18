# RPA — Roast Profile Analyzator

Standalone Roast Profile database for Artisan `.alog` files.

## Scope V0.2.0

- Import one Artisan `.alog`
- Save raw roast log locally
- Store BT, ET, RoR, milestones, and Artisan events
- Search roast profile database
- Open saved profiles and charts
- Record Agtron, purpose, profile version, status, and notes

AI analysis is intentionally not included in this release. The first phase is database utilization.

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Data policy

The repository may be public, but operational data stays local.

The following are ignored by Git:

- `data/rpa_roast_profiles.db`
- `data/artisan_raw/*.alog`
- other local files inside `data/`

Do not manually remove `.gitignore`.


## Legacy database migration

Copy the supplied `rpa_roast_profiles.db` into the local `data` folder. Do not upload this database to GitHub.
