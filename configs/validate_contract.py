"""
Validador del contrato de datos Fudo → Skalar.

Uso:
    uv run python configs/validate_contract.py datos_export.csv
    uv run python configs/validate_contract.py datos_export.parquet

Lee data_contract.yaml y valida que los datos cumplan con el esquema.
Retorna exit code 0 si pasa, 1 si hay errores.
"""

import sys
from pathlib import Path

import pandas as pd
import yaml

CONTRACT_PATH = Path(__file__).parent / "data_contract.yaml"

TYPE_MAP = {
    "STRING": {"object", "str", "string"},
    "DATE": {"datetime64[ns]", "datetime64[us]", "date"},
    "FLOAT64": {"float64"},
}


def load_data(path: str) -> pd.DataFrame:
    p = Path(path)
    if p.suffix == ".csv":
        return pd.read_csv(p)
    if p.suffix in (".parquet", ".pq"):
        return pd.read_parquet(p)
    raise ValueError(f"Formato no soportado: {p.suffix}")


def validate(df: pd.DataFrame, contract: dict) -> list[dict]:
    errors = []

    contract_cols = {c["name"]: c for c in contract["columns"]}

    # Columnas faltantes
    for name, spec in contract_cols.items():
        if name not in df.columns:
            if not spec.get("nullable", True):
                errors.append({"column": name, "check": "presencia", "detail": "Columna NOT NULL ausente"})
            continue

        col = df[name]

        # Tipo
        actual = str(col.dtype)
        expected = TYPE_MAP.get(spec["type"], {spec["type"].lower()})
        if actual not in expected:
            errors.append({
                "column": name,
                "check": "tipo",
                "detail": f"Esperado {spec['type']} ({expected}), recibido {actual}",
            })

        # Nulabilidad
        if not spec.get("nullable", True):
            n_null = int(col.isnull().sum())
            if n_null > 0:
                errors.append({
                    "column": name,
                    "check": "nulabilidad",
                    "detail": f"{n_null} valores nulos en columna NOT NULL",
                })

        # Valores válidos
        if "valid_values" in spec:
            allowed = set(spec["valid_values"])
            actual_vals = set(col.dropna().unique())
            unexpected = actual_vals - allowed
            if unexpected:
                errors.append({
                    "column": name,
                    "check": "dominio",
                    "detail": f"Valores no permitidos: {sorted(unexpected)}",
                })

        # Rango mínimo
        if "min" in spec and pd.api.types.is_numeric_dtype(col):
            below = int((col < spec["min"]).sum())
            if below > 0:
                errors.append({
                    "column": name,
                    "check": "rango",
                    "detail": f"{below} valores por debajo del mínimo ({spec['min']})",
                })

    return errors


def main():
    if len(sys.argv) < 2:
        print(f"Uso: python {sys.argv[0]} <archivo.csv|archivo.parquet>")
        sys.exit(2)

    with open(CONTRACT_PATH) as f:
        contract = yaml.safe_load(f)

    df = load_data(sys.argv[1])
    errors = validate(df, contract)

    if not errors:
        print("✓ Todos los checks pasaron.")
        sys.exit(0)

    print(f"✗ {len(errors)} error(es) encontrado(s):\n")
    for e in errors:
        print(f"  [{e['column']}] {e['check']}: {e['detail']}")
    sys.exit(1)


if __name__ == "__main__":
    main()
