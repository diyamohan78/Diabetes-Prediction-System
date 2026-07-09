# ==========================================================
# Diabetes Prediction System
# Database Helper (SQLite)
# ==========================================================
# Lightweight local storage for patient details and prediction
# results. Uses SQLite so no external database server is needed.
# ==========================================================

import sqlite3
from contextlib import contextmanager
from datetime import datetime

import pandas as pd

DB_PATH = "patient_records.db"

TABLE_NAME = "patient_predictions"

_CREATE_TABLE_SQL = f"""
CREATE TABLE IF NOT EXISTS {TABLE_NAME} (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    full_name           TEXT NOT NULL,
    contact_number      TEXT,
    pregnancies         INTEGER,
    glucose             REAL,
    blood_pressure      REAL,
    skin_thickness      REAL,
    insulin             REAL,
    bmi                 REAL,
    diabetes_pedigree   REAL,
    age                 INTEGER,
    prediction          TEXT NOT NULL,
    probability_diabetic REAL,
    probability_non_diabetic REAL,
    created_at          TEXT NOT NULL
);
"""


@contextmanager
def _get_connection():
    """Yield a SQLite connection, always closing it afterward."""
    conn = sqlite3.connect(DB_PATH)
    try:
        yield conn
    finally:
        conn.close()


def init_db() -> None:
    """Create the patient_predictions table if it doesn't already exist."""
    with _get_connection() as conn:
        conn.execute(_CREATE_TABLE_SQL)
        conn.commit()


def insert_record(
    full_name: str,
    contact_number: str,
    pregnancies: int,
    glucose: float,
    blood_pressure: float,
    skin_thickness: float,
    insulin: float,
    bmi: float,
    diabetes_pedigree: float,
    age: int,
    prediction: str,
    probability_diabetic: float,
    probability_non_diabetic: float,
) -> int:
    """
    Insert one patient record + prediction result into the database.
    Returns the new row's id.
    """
    with _get_connection() as conn:
        cursor = conn.execute(
            f"""
            INSERT INTO {TABLE_NAME} (
                full_name, contact_number, pregnancies, glucose,
                blood_pressure, skin_thickness, insulin, bmi,
                diabetes_pedigree, age, prediction,
                probability_diabetic, probability_non_diabetic, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                full_name.strip(),
                contact_number.strip() if contact_number else None,
                pregnancies,
                glucose,
                blood_pressure,
                skin_thickness,
                insulin,
                bmi,
                diabetes_pedigree,
                age,
                prediction,
                probability_diabetic,
                probability_non_diabetic,
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            ),
        )
        conn.commit()
        return cursor.lastrowid


def fetch_all_records() -> pd.DataFrame:
    """Return all saved patient records, most recent first."""
    with _get_connection() as conn:
        return pd.read_sql_query(
            f"SELECT * FROM {TABLE_NAME} ORDER BY id DESC", conn
        )


def delete_record(record_id: int) -> None:
    """Delete a single record by its id."""
    with _get_connection() as conn:
        conn.execute(f"DELETE FROM {TABLE_NAME} WHERE id = ?", (record_id,))
        conn.commit()


def delete_all_records() -> None:
    """Delete every saved record (used by the 'clear history' action)."""
    with _get_connection() as conn:
        conn.execute(f"DELETE FROM {TABLE_NAME}")
        conn.commit()