from __future__ import annotations

import ast
import hashlib
import json
import sqlite3
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

APP_VERSION = "0.3.0"
APP_DIR = Path(__file__).resolve().parent
DATA_DIR = APP_DIR / "data"
RAW_DIR = DATA_DIR / "artisan_raw"
KNOWLEDGE_DIR = DATA_DIR / "knowledge_images"
DB_PATH = DATA_DIR / "rpa_roast_profiles.db"
RAW_DIR.mkdir(parents=True, exist_ok=True)
KNOWLEDGE_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class ParsedRoast:
    raw: dict[str, Any]
    file_name: str
    checksum: str
    summary: dict[str, Any]
    curves: pd.DataFrame
    events: pd.DataFrame


def db() -> sqlite3.Connection:
    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA foreign_keys = ON")
    return con


def init_db() -> None:
    with db() as con:
        con.executescript(
            """
            CREATE TABLE IF NOT EXISTS green_beans (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                bean_name TEXT NOT NULL,
                species TEXT,
                origin TEXT,
                process TEXT,
                supplier TEXT DEFAULT 'Unknown',
                lot TEXT,
                variety TEXT,
                density REAL,
                moisture REAL,
                initial_weight_g REAL,
                current_stock_g REAL,
                purchase_price REAL,
                selling_price REAL,
                status TEXT DEFAULT 'Active',
                UNIQUE(bean_name, process, supplier, lot)
            );

            CREATE TABLE IF NOT EXISTS roast_sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                roast_id TEXT UNIQUE NOT NULL,
                bean_id INTEGER,
                title TEXT,
                roast_date TEXT,
                roast_time TEXT,
                operator TEXT,
                roaster TEXT,
                artisan_version TEXT,
                roast_purpose TEXT,
                blend_project TEXT,
                profile_version TEXT,
                status TEXT DEFAULT 'Trial',
                drum_speed_rpm REAL DEFAULT 90,
                green_weight_g REAL,
                roasted_weight_g REAL,
                weight_loss_pct REAL,
                yield_pct REAL,
                density REAL,
                moisture REAL,
                agtron REAL,
                notes TEXT,
                data_quality TEXT,
                parser_version TEXT DEFAULT '0.3',
                created_at TEXT NOT NULL,
                FOREIGN KEY(bean_id) REFERENCES green_beans(id)
            );

            CREATE TABLE IF NOT EXISTS artisan_files (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                roast_session_id INTEGER NOT NULL,
                original_name TEXT,
                source_path TEXT,
                stored_path TEXT,
                checksum_sha256 TEXT UNIQUE,
                raw_json TEXT,
                imported_at TEXT NOT NULL,
                FOREIGN KEY(roast_session_id) REFERENCES roast_sessions(id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS roast_curve_points (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                roast_session_id INTEGER NOT NULL,
                time_s REAL,
                bt REAL,
                et REAL,
                ror_bt REAL,
                ror_et REAL,
                FOREIGN KEY(roast_session_id) REFERENCES roast_sessions(id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS roast_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                roast_session_id INTEGER NOT NULL,
                event_name TEXT,
                time_s REAL,
                bt REAL,
                et REAL,
                gas REAL,
                airflow REAL,
                drum REAL,
                source TEXT,
                FOREIGN KEY(roast_session_id) REFERENCES roast_sessions(id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS roast_milestones (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                roast_session_id INTEGER NOT NULL,
                milestone TEXT,
                time_s REAL,
                bt REAL,
                et REAL,
                FOREIGN KEY(roast_session_id) REFERENCES roast_sessions(id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS coffee_knowledge (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT NOT NULL,
                category TEXT NOT NULL,
                tags TEXT,
                source TEXT,
                notes TEXT,
                image_path TEXT NOT NULL,
                created_at TEXT NOT NULL
            );
            """
        )
        # Compatibility upgrade for the stable Roast Log V0.4.4 database.
        columns = {row[1] for row in con.execute("PRAGMA table_info(roast_sessions)")}
        if "status" not in columns:
            con.execute("ALTER TABLE roast_sessions ADD COLUMN status TEXT DEFAULT 'Trial'")
        con.execute("UPDATE roast_sessions SET status = 'Trial' WHERE status IS NULL OR status = ''")


def safe_num(value: Any) -> float | None:
    try:
        if value in (None, "", -1):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def parse_alog(file_name: str, payload: bytes) -> ParsedRoast:
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError:
        text = payload.decode("latin-1")

    try:
        raw = ast.literal_eval(text)
    except Exception as exc:
        raise ValueError(f"File .alog tidak dapat dibaca: {exc}") from exc

    if not isinstance(raw, dict):
        raise ValueError("Isi file bukan dictionary Artisan.")

    timex = raw.get("timex") or []
    et = raw.get("temp1") or []
    bt = raw.get("temp2") or []
    if not timex or not bt:
        raise ValueError("Kurva waktu atau BT tidak ditemukan.")

    timeindex = raw.get("timeindex") or [0]
    charge_idx = int(timeindex[0] or 0)
    charge_t = float(timex[charge_idx]) if charge_idx < len(timex) else 0.0
    n = min(len(timex), len(bt), len(et) if et else len(bt))

    rows = []
    for i in range(charge_idx, n):
        rows.append(
            {
                "time_s": round(float(timex[i]) - charge_t, 3),
                "bt": safe_num(bt[i]),
                "et": safe_num(et[i]) if et else None,
            }
        )

    curves = pd.DataFrame(rows)
    if curves.empty:
        raise ValueError("Kurva roast kosong.")

    curves["ror_bt"] = curves["bt"].diff() / curves["time_s"].diff() * 60
    curves.replace([float("inf"), float("-inf")], pd.NA, inplace=True)

    computed = raw.get("computed") or {}
    weight = raw.get("weight") or [None, None, "g"]

    green_weight = safe_num(computed.get("weightin")) or safe_num(weight[0] if len(weight) > 0 else None)
    roasted_weight = safe_num(computed.get("weightout")) or safe_num(weight[1] if len(weight) > 1 else None)
    loss = safe_num(computed.get("weight_loss"))
    if loss is None and green_weight and roasted_weight:
        loss = (green_weight - roasted_weight) / green_weight * 100

    summary = {
        "title": raw.get("title") or Path(file_name).stem,
        "roast_date": raw.get("roastisodate") or raw.get("roastdate"),
        "roast_time": raw.get("roasttime"),
        "operator": raw.get("operator"),
        "roaster": raw.get("roastertype") or raw.get("machinesetup"),
        "artisan_version": raw.get("recording_version") or raw.get("version"),
        "drum_speed": safe_num(raw.get("drumspeed")) or 90.0,
        "green_weight_g": green_weight,
        "roasted_weight_g": roasted_weight,
        "weight_loss_pct": loss,
        "yield_pct": 100 - loss if loss is not None else None,
        "density": safe_num(computed.get("set_density")) or safe_num((raw.get("density") or [None])[0]),
        "moisture": safe_num(computed.get("moisture_greens")) or safe_num(raw.get("moisture_greens")),
        "charge_bt": safe_num(computed.get("CHARGE_BT")),
        "charge_et": safe_num(computed.get("CHARGE_ET")),
        "tp_time": safe_num(computed.get("TP_time")),
        "tp_bt": safe_num(computed.get("TP_BT")),
        "tp_et": safe_num(computed.get("TP_ET")),
        "dry_time": safe_num(computed.get("DRY_time")),
        "dry_bt": safe_num(computed.get("DRY_BT")),
        "dry_et": safe_num(computed.get("DRY_ET")),
        "fc_time": safe_num(computed.get("FCs_time")),
        "fc_bt": safe_num(computed.get("FCs_BT")),
        "fc_et": safe_num(computed.get("FCs_ET")),
        "drop_time": safe_num(computed.get("DROP_time")),
        "drop_bt": safe_num(computed.get("DROP_BT")),
        "drop_et": safe_num(computed.get("DROP_ET")),
        "total_time": safe_num(computed.get("totaltime")),
        "drying_time": safe_num(computed.get("dryphasetime")),
        "maillard_time": safe_num(computed.get("midphasetime")),
        "development_time": safe_num(computed.get("finishphasetime")),
    }
    summary["dtr_pct"] = (
        summary["development_time"] / summary["total_time"] * 100
        if summary["development_time"] and summary["total_time"]
        else None
    )

    event_types = raw.get("etypes") or ["Air", "Drum", "Damper", "Burner", "--"]
    event_indexes = raw.get("specialevents") or []
    event_type_indexes = raw.get("specialeventstype") or []
    event_values = raw.get("specialeventsvalue") or []

    event_rows = []
    for idx, typ, value in zip(event_indexes, event_type_indexes, event_values):
        if idx >= len(timex):
            continue
        name = event_types[typ] if isinstance(typ, int) and 0 <= typ < len(event_types) else f"Type {typ}"
        event_rows.append(
            {
                "event_name": name,
                "time_s": round(float(timex[idx]) - charge_t, 3),
                "bt": safe_num(bt[idx]) if idx < len(bt) else None,
                "et": safe_num(et[idx]) if idx < len(et) else None,
                "gas": safe_num(value) if str(name).lower() in {"burner", "gas"} else None,
                "airflow": safe_num(value) if str(name).lower() in {"air", "airflow", "fan"} else None,
                "drum": safe_num(value) if str(name).lower() in {"drum", "drumspeed"} else None,
                "source": "Artisan",
            }
        )

    events = pd.DataFrame(
        event_rows,
        columns=["event_name", "time_s", "bt", "et", "gas", "airflow", "drum", "source"],
    )

    return ParsedRoast(
        raw=raw,
        file_name=file_name,
        checksum=sha256_bytes(payload),
        summary=summary,
        curves=curves,
        events=events,
    )


def fmt_time(seconds: float | None) -> str:
    if seconds is None or pd.isna(seconds):
        return "—"
    seconds = int(round(seconds))
    return f"{seconds // 60}:{seconds % 60:02d}"


def existing_checksum(checksum: str) -> sqlite3.Row | None:
    with db() as con:
        return con.execute(
            """
            SELECT rs.roast_id, rs.title
            FROM artisan_files af
            JOIN roast_sessions rs ON rs.id = af.roast_session_id
            WHERE af.checksum_sha256 = ?
            """,
            (checksum,),
        ).fetchone()


def make_roast_id(roast_date: str | None, con: sqlite3.Connection) -> str:
    date_code = (roast_date or datetime.now().date().isoformat()).replace("-", "")
    prefix = f"RPA-{date_code}"
    count = con.execute(
        "SELECT COUNT(*) FROM roast_sessions WHERE roast_id LIKE ?",
        (prefix + "%",),
    ).fetchone()[0]
    return f"{prefix}-{count + 1:03d}"


def ensure_bean(
    con: sqlite3.Connection,
    bean_name: str,
    species: str,
    origin: str,
    process: str,
    supplier: str,
    lot: str,
    density: float | None,
    moisture: float | None,
) -> int:
    row = con.execute(
        """
        SELECT id FROM green_beans
        WHERE bean_name = ?
          AND COALESCE(process, '') = ?
          AND COALESCE(supplier, '') = ?
          AND COALESCE(lot, '') = ?
        """,
        (bean_name, process, supplier, lot),
    ).fetchone()

    if row:
        return int(row["id"])

    cur = con.execute(
        """
        INSERT INTO green_beans(
            bean_name, species, origin, process, supplier, lot, density, moisture
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (bean_name, species, origin, process, supplier, lot, density, moisture),
    )
    return int(cur.lastrowid)


def save_roast(parsed: ParsedRoast, form: dict[str, Any]) -> str:
    duplicate = existing_checksum(parsed.checksum)
    if duplicate:
        raise ValueError(
            f"File sudah pernah disimpan sebagai {duplicate['roast_id']} — {duplicate['title']}"
        )

    with db() as con:
        bean_id = ensure_bean(
            con,
            form["bean_name"],
            form["species"],
            form["origin"],
            form["process"],
            form["supplier"],
            form["lot"],
            parsed.summary.get("density"),
            parsed.summary.get("moisture"),
        )

        roast_id = make_roast_id(parsed.summary.get("roast_date"), con)
        cur = con.execute(
            """
            INSERT INTO roast_sessions(
                roast_id, bean_id, title, roast_date, roast_time, operator, roaster,
                artisan_version, roast_purpose, profile_version, status,
                drum_speed_rpm, green_weight_g, roasted_weight_g, weight_loss_pct,
                yield_pct, agtron, notes, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                roast_id,
                bean_id,
                parsed.summary.get("title"),
                parsed.summary.get("roast_date"),
                parsed.summary.get("roast_time"),
                parsed.summary.get("operator"),
                parsed.summary.get("roaster"),
                parsed.summary.get("artisan_version"),
                form["purpose"],
                form["profile_version"],
                form["status"],
                form["drum_speed"],
                parsed.summary.get("green_weight_g"),
                parsed.summary.get("roasted_weight_g"),
                parsed.summary.get("weight_loss_pct"),
                parsed.summary.get("yield_pct"),
                form["agtron"],
                form["notes"],
                datetime.now().isoformat(timespec="seconds"),
            ),
        )
        session_id = int(cur.lastrowid)

        stored_path = RAW_DIR / f"{roast_id}_{Path(parsed.file_name).name}"
        stored_path.write_text(repr(parsed.raw), encoding="utf-8")

        con.execute(
            """
            INSERT INTO artisan_files(
                roast_session_id, original_name, source_path, stored_path, checksum_sha256,
                raw_json, imported_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                session_id,
                parsed.file_name,
                "Imported through RPA",
                str(stored_path),
                parsed.checksum,
                json.dumps(parsed.raw, default=str),
                datetime.now().isoformat(timespec="seconds"),
            ),
        )

        con.executemany(
            """
            INSERT INTO roast_curve_points(
                roast_session_id, time_s, bt, et, ror_bt, ror_et
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    session_id,
                    row.time_s,
                    row.bt,
                    row.et,
                    None if pd.isna(row.ror_bt) else row.ror_bt,
                    None,
                )
                for row in parsed.curves.itertuples(index=False)
            ],
        )

        if not parsed.events.empty:
            con.executemany(
                """
                INSERT INTO roast_events(
                    roast_session_id, event_name, time_s, bt, et, gas, airflow, drum, source
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        session_id,
                        row.event_name,
                        row.time_s,
                        row.bt,
                        row.et,
                        row.gas,
                        row.airflow,
                        row.drum,
                        row.source,
                    )
                    for row in parsed.events.itertuples(index=False)
                ],
            )

        milestones = [
            ("Charge", 0, parsed.summary.get("charge_bt"), parsed.summary.get("charge_et")),
            ("Turning Point", parsed.summary.get("tp_time"), parsed.summary.get("tp_bt"), parsed.summary.get("tp_et")),
            ("Dry End", parsed.summary.get("dry_time"), parsed.summary.get("dry_bt"), parsed.summary.get("dry_et")),
            ("First Crack", parsed.summary.get("fc_time"), parsed.summary.get("fc_bt"), parsed.summary.get("fc_et")),
            ("Drop", parsed.summary.get("drop_time"), parsed.summary.get("drop_bt"), parsed.summary.get("drop_et")),
        ]
        con.executemany(
            """
            INSERT INTO roast_milestones(
                roast_session_id, milestone, time_s, bt, et
            ) VALUES (?, ?, ?, ?, ?)
            """,
            [(session_id, *row) for row in milestones if row[1] is not None],
        )

    return roast_id


def roast_chart(curves: pd.DataFrame, milestones: pd.DataFrame | None = None) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=curves["time_s"], y=curves["bt"], name="BT", mode="lines"))
    if "et" in curves and curves["et"].notna().any():
        fig.add_trace(go.Scatter(x=curves["time_s"], y=curves["et"], name="ET", mode="lines"))

    if milestones is not None and not milestones.empty:
        for row in milestones.itertuples(index=False):
            if pd.notna(row.time_s) and row.milestone in {"Dry End", "First Crack", "Drop"}:
                fig.add_vline(x=float(row.time_s), line_dash="dash", opacity=0.5)
                fig.add_annotation(
                    x=float(row.time_s),
                    y=1,
                    yref="paper",
                    text=row.milestone,
                    showarrow=False,
                )

    fig.update_layout(
        height=500,
        xaxis_title="Time from Charge (s)",
        yaxis_title="Temperature (°C)",
        margin=dict(l=20, r=20, t=30, b=20),
    )
    return fig


def render_home() -> None:
    with db() as con:
        total = con.execute("SELECT COUNT(*) FROM roast_sessions").fetchone()[0]
        arabica = con.execute(
            """
            SELECT COUNT(*) FROM roast_sessions rs
            JOIN green_beans gb ON gb.id = rs.bean_id
            WHERE gb.species = 'Arabica'
            """
        ).fetchone()[0]
        robusta = con.execute(
            """
            SELECT COUNT(*) FROM roast_sessions rs
            JOIN green_beans gb ON gb.id = rs.bean_id
            WHERE gb.species = 'Robusta'
            """
        ).fetchone()[0]
        locked = con.execute(
            "SELECT COUNT(*) FROM roast_sessions WHERE status = 'Locked'"
        ).fetchone()[0]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Roast Logs", total)
    c2.metric("Arabica", arabica)
    c3.metric("Robusta", robusta)
    c4.metric("Locked Profiles", locked)

    st.markdown("### RPA Database")
    st.write(
        "Tahap pertama RPA berfungsi sebagai database roast profile mandiri. "
        "Data aplikasi dan source code terpisah: source code berada di GitHub, "
        "sedangkan database serta file `.alog` tetap lokal di laptop."
    )


def render_import() -> None:
    st.subheader("Import Artisan Roast Log")
    uploaded = st.file_uploader("Pilih file Artisan `.alog`", type=["alog"])
    if not uploaded:
        st.info("Upload satu file `.alog` untuk preview dan penyimpanan.")
        return

    try:
        parsed = parse_alog(uploaded.name, uploaded.getvalue())
    except Exception as exc:
        st.error(str(exc))
        return

    duplicate = existing_checksum(parsed.checksum)
    if duplicate:
        st.warning(
            f"File ini sudah ada sebagai {duplicate['roast_id']} — {duplicate['title']}."
        )

    st.plotly_chart(roast_chart(parsed.curves), width="stretch")

    s = parsed.summary
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Dry End", fmt_time(s.get("dry_time")))
    m2.metric("First Crack", fmt_time(s.get("fc_time")))
    m3.metric("Drop", fmt_time(s.get("drop_time")))
    m4.metric("DTR", f"{s['dtr_pct']:.1f}%" if s.get("dtr_pct") is not None else "—")

    with st.form("save_roast_form"):
        left, right = st.columns(2)
        with left:
            bean_name = st.text_input("Bean Name", value=s.get("title") or Path(uploaded.name).stem)
            species = st.selectbox("Species", ["Arabica", "Robusta", "Unknown"])
            origin = st.text_input("Origin")
            process = st.text_input("Process")
            supplier = st.text_input("Supplier", value="Unknown")
            lot = st.text_input("Lot")
        with right:
            purpose = st.selectbox(
                "Roast Purpose",
                ["Filter", "Espresso", "Latte", "Tubruk", "Blend Component", "Experimental", "Other"],
            )
            profile_version = st.text_input("Profile Version", value=s.get("title") or "")
            status = st.selectbox("Profile Status", ["Trial", "Evaluation", "Approved", "Locked", "Archived"])
            drum_speed = st.number_input("Drum Speed (rpm)", min_value=0.0, value=float(s.get("drum_speed") or 90.0))
            agtron = st.number_input("Agtron", min_value=0.0, value=0.0, step=1.0)
            notes = st.text_area("Notes")

        submitted = st.form_submit_button(
            "Save Roast",
            type="primary",
            disabled=bool(duplicate),
        )

    if submitted:
        if not bean_name.strip():
            st.error("Bean Name wajib diisi.")
            return
        try:
            roast_id = save_roast(
                parsed,
                {
                    "bean_name": bean_name.strip(),
                    "species": species,
                    "origin": origin.strip(),
                    "process": process.strip(),
                    "supplier": supplier.strip() or "Unknown",
                    "lot": lot.strip(),
                    "purpose": purpose,
                    "profile_version": profile_version.strip(),
                    "status": status,
                    "drum_speed": drum_speed,
                    "agtron": None if agtron == 0 else agtron,
                    "notes": notes.strip(),
                },
            )
            st.success(f"Roast berhasil disimpan: {roast_id}")
        except Exception as exc:
            st.error(str(exc))


def load_database() -> pd.DataFrame:
    with db() as con:
        return pd.read_sql_query(
            """
            SELECT
                rs.roast_id,
                rs.roast_date,
                rs.title,
                gb.bean_name,
                gb.species,
                gb.origin,
                gb.process,
                rs.roast_purpose,
                rs.profile_version,
                rs.status,
                rs.green_weight_g,
                rs.weight_loss_pct,
                rs.agtron
            FROM roast_sessions rs
            LEFT JOIN green_beans gb ON gb.id = rs.bean_id
            ORDER BY rs.id DESC
            """,
            con,
        )


def render_database() -> None:
    st.subheader("Roast Profile Database")
    df = load_database()
    if df.empty:
        st.info("Database masih kosong.")
        return

    query = st.text_input("Search", placeholder="Bean, origin, process, purpose, version, status...")
    filtered = df.copy()
    if query.strip():
        mask = filtered.astype(str).apply(
            lambda col: col.str.contains(query.strip(), case=False, na=False)
        ).any(axis=1)
        filtered = filtered[mask]

    st.dataframe(filtered, hide_index=True, width="stretch")

    roast_ids = filtered["roast_id"].tolist()
    if not roast_ids:
        return

    selected = st.selectbox("Open Roast Profile", roast_ids)
    render_roast_detail(selected)


def render_roast_detail(roast_id: str) -> None:
    with db() as con:
        row = con.execute(
            """
            SELECT
                rs.*,
                gb.bean_name,
                gb.species,
                gb.origin,
                gb.process,
                gb.supplier,
                gb.lot
            FROM roast_sessions rs
            LEFT JOIN green_beans gb ON gb.id = rs.bean_id
            WHERE rs.roast_id = ?
            """,
            (roast_id,),
        ).fetchone()
        if not row:
            st.error("Roast tidak ditemukan.")
            return

        curves = pd.read_sql_query(
            """
            SELECT time_s, bt, et, ror_bt
            FROM roast_curve_points
            WHERE roast_session_id = ?
            ORDER BY time_s
            """,
            con,
            params=(row["id"],),
        )
        milestones = pd.read_sql_query(
            """
            SELECT milestone, time_s, bt, et
            FROM roast_milestones
            WHERE roast_session_id = ?
            ORDER BY time_s
            """,
            con,
            params=(row["id"],),
        )
        events = pd.read_sql_query(
            """
            SELECT event_name, time_s, bt, et, gas, airflow, drum, source
            FROM roast_events
            WHERE roast_session_id = ?
            ORDER BY time_s
            """,
            con,
            params=(row["id"],),
        )

    st.markdown(f"### {row['roast_id']} — {row['bean_name']}")
    st.caption(
        f"{row['species'] or 'Unknown'} · {row['origin'] or 'Unknown'} · "
        f"{row['process'] or 'Unknown'} · {row['roast_purpose'] or 'Unknown'} · "
        f"{row['status'] or 'Trial'}"
    )

    if not curves.empty:
        st.plotly_chart(roast_chart(curves, milestones), width="stretch")

    left, right = st.columns(2)
    with left:
        st.markdown("#### Milestones")
        if milestones.empty:
            st.caption("Tidak ada milestone.")
        else:
            view = milestones.copy()
            view["Time"] = view["time_s"].apply(fmt_time)
            st.dataframe(
                view.rename(columns={"milestone": "Milestone", "bt": "BT", "et": "ET"})[
                    ["Milestone", "Time", "BT", "ET"]
                ],
                hide_index=True,
                width="stretch",
            )
    with right:
        st.markdown("#### Physical Result")
        physical = pd.DataFrame(
            [
                ["Green Weight", row["green_weight_g"], "g"],
                ["Roasted Weight", row["roasted_weight_g"], "g"],
                ["Weight Loss", row["weight_loss_pct"], "%"],
                ["Yield", row["yield_pct"], "%"],
                ["Agtron", row["agtron"], ""],
                ["Drum Speed", row["drum_speed_rpm"], "rpm"],
            ],
            columns=["Parameter", "Value", "Unit"],
        )
        st.dataframe(physical, hide_index=True, width="stretch")

    st.markdown("#### Artisan Events")
    if events.empty:
        st.caption("Tidak ada event tersimpan.")
    else:
        st.dataframe(events, hide_index=True, width="stretch")

    st.markdown("#### Notes")
    st.write(row["notes"] or "—")



def save_knowledge_image(uploaded_file: Any) -> str:
    suffix = Path(uploaded_file.name).suffix.lower() or ".png"
    safe_stem = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in Path(uploaded_file.name).stem)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    destination = KNOWLEDGE_DIR / f"{stamp}_{safe_stem}{suffix}"
    destination.write_bytes(uploaded_file.getvalue())
    return str(destination.relative_to(APP_DIR))


def render_knowledge_library() -> None:
    st.subheader("Coffee Knowledge Library")
    st.caption("Koleksi ilmu kopi lokal: gambar, kategori, sumber, tag, dan catatan.")

    with st.expander("➕ Tambah Knowledge Baru", expanded=False):
        with st.form("knowledge_form", clear_on_submit=True):
            left, right = st.columns(2)
            with left:
                title = st.text_input("Judul")
                category = st.selectbox(
                    "Kategori",
                    [
                        "Coffee Processing",
                        "Green Bean & Origin",
                        "Roasting",
                        "Roast Defects",
                        "Sensory & Cupping",
                        "Brewing",
                        "Machine & Maintenance",
                        "Other Coffee Knowledge",
                    ],
                )
                tags = st.text_input("Tags", placeholder="washed, process, acidity")
            with right:
                source = st.text_input("Sumber")
                image = st.file_uploader("Gambar", type=["png", "jpg", "jpeg", "webp"])
                notes = st.text_area("Catatan pribadi")
            submit = st.form_submit_button("Save Knowledge", type="primary")

        if submit:
            if not title.strip() or image is None:
                st.error("Judul dan gambar wajib diisi.")
            else:
                try:
                    image_path = save_knowledge_image(image)
                    with db() as con:
                        con.execute(
                            """
                            INSERT INTO coffee_knowledge(
                                title, category, tags, source, notes, image_path, created_at
                            ) VALUES (?, ?, ?, ?, ?, ?, ?)
                            """,
                            (
                                title.strip(), category, tags.strip(), source.strip(), notes.strip(),
                                image_path, datetime.now().isoformat(timespec="seconds"),
                            ),
                        )
                    st.success(f"Knowledge '{title.strip()}' berhasil disimpan.")
                    st.rerun()
                except Exception as exc:
                    st.error(f"Gagal menyimpan knowledge: {exc}")

    with db() as con:
        records = pd.read_sql_query(
            """
            SELECT id, title, category, tags, source, notes, image_path, created_at
            FROM coffee_knowledge ORDER BY id
            """,
            con,
        )

    if records.empty:
        st.info("Coffee Knowledge Library masih kosong.")
        return

    q1, q2 = st.columns([2, 1])
    query = q1.text_input("Search Knowledge", placeholder="Judul, kategori, tag, sumber...")
    categories = ["All"] + sorted(records["category"].dropna().unique().tolist())
    selected_category = q2.selectbox("Category", categories)

    filtered = records.copy()
    if selected_category != "All":
        filtered = filtered[filtered["category"] == selected_category]
    if query.strip():
        haystack = filtered[["title", "category", "tags", "source", "notes"]].fillna("").astype(str)
        mask = haystack.apply(lambda col: col.str.contains(query.strip(), case=False, na=False)).any(axis=1)
        filtered = filtered[mask]

    st.caption(f"{len(filtered)} knowledge item")
    if filtered.empty:
        st.warning("Tidak ada knowledge yang cocok.")
        return

    for row in filtered.itertuples(index=False):
        with st.expander(f"{row.title}  ·  {row.category}", expanded=False):
            image_file = APP_DIR / row.image_path
            if image_file.exists():
                st.image(str(image_file), width="stretch")
            else:
                st.warning(f"File gambar tidak ditemukan: {row.image_path}")
            meta = []
            if row.tags: meta.append(f"**Tags:** {row.tags}")
            if row.source: meta.append(f"**Sumber:** {row.source}")
            if meta: st.markdown("  \n".join(meta))
            if row.notes:
                st.markdown("**Catatan:**")
                st.write(row.notes)
            if st.button("Hapus dari Library", key=f"delete_knowledge_{row.id}"):
                try:
                    with db() as con:
                        con.execute("DELETE FROM coffee_knowledge WHERE id = ?", (int(row.id),))
                    if image_file.exists():
                        image_file.unlink()
                    st.success("Knowledge dihapus.")
                    st.rerun()
                except Exception as exc:
                    st.error(f"Gagal menghapus: {exc}")


def render_release_history() -> None:
    st.subheader("Release History")
    st.markdown(
        """
**V0.3.0 — Coffee Knowledge Library**

- Adds a local Coffee Knowledge Library inside RPA.
- Saves image, title, category, tags, source, and personal notes.
- Includes search and category filters.
- Supports adding and deleting knowledge items directly from the application.
- Seeds seven coffee-processing infographics as the initial knowledge database.
- Knowledge images remain local and are excluded from GitHub.

**V0.2.0 — Legacy Database Migration**

- Compatible with the stable Roast Log V0.4.4 database.
- Migrates and opens 561 existing roast profiles without manual re-import.
- Preserves raw Artisan files, curve points, milestones, events, and bean records.
- Adds profile status safely without overwriting the original database.

**V0.1.0 — RPA Database Foundation**

- RPA berdiri sebagai project mandiri, terpisah dari CIS.
- Import satu file Artisan `.alog`.
- Penyimpanan raw `.alog`, kurva BT/ET/RoR, milestone, dan event.
- Roast Profile Database dengan search.
- Pembukaan kembali roast detail dan grafik.
- Profile status: Trial, Evaluation, Approved, Locked, Archived.
- Database dan raw roast tetap lokal serta tidak masuk GitHub.
- Belum ada AI analysis; fokus tahap ini murni database.
"""
    )


def main() -> None:
    st.set_page_config(
        page_title="RPA — Roast Profile Analyzator",
        page_icon="🔥",
        layout="wide",
    )
    init_db()

    st.title("🔥 RPA")
    st.caption(f"Roast Profile Analyzator · Local Database V{APP_VERSION}")

    page = st.sidebar.radio(
        "Menu",
        ["Home", "Import Roast", "Roast Database", "Coffee Knowledge", "Release History"],
    )

    if page == "Home":
        render_home()
    elif page == "Import Roast":
        render_import()
    elif page == "Roast Database":
        render_database()
    elif page == "Coffee Knowledge":
        render_knowledge_library()
    else:
        render_release_history()


if __name__ == "__main__":
    main()
