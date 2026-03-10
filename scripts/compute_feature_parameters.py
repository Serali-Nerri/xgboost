#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path


OUTPUT_COLUMNS = [
    "b (mm)",
    "h (mm)",
    "r0 (mm)",
    "t (mm)",
    "R (%)",
    "fy (MPa)",
    "fc (MPa)",
    "L (mm)",
    "e1 (mm)",
    "e2 (mm)",
    "Nexp (kN)",
    "r0/h",
    "b/t",
    "Ac (mm^2)",
    "As (mm^2)",
    "Re (mm)",
    "te (mm)",
    "ke",
    "xi",
    "sigma_re (MPa)",
    "lambda",
    "lambda_bar",
    "e/h",
    "e1/e2",
    "e_bar",
    "Npl (kN)",
    "psi",
    "b/h",
    "L/h",
    "axial_flag",
    "section_family",
]

INPUT_ALIASES = {
    "b (mm)": ("b (mm)",),
    "h (mm)": ("h (mm)",),
    "r0 (mm)": ("r0 (mm)",),
    "t (mm)": ("t (mm)",),
    "R (%)": ("R (%)", "R再生骨料取代率(%)"),
    "fy (MPa)": ("fy (MPa)",),
    "fc (MPa)": ("fc (MPa)",),
    "L (mm)": ("L (mm)",),
    "e1 (mm)": ("e1 (mm)",),
    "e2 (mm)": ("e2 (mm)",),
    "Nexp (kN)": ("Nexp (kN)",),
}
OPTIONAL_INPUT_COLUMNS = {"Nexp (kN)"}

STEEL_ELASTIC_MODULUS = 206000.0
PLACEHOLDER_VALUES = {"", "-", "--", "—", "–", "NA", "N/A", "na", "nan", "NaN"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute CFST feature parameters from a raw CSV file."
    )
    parser.add_argument(
        "--input",
        required=True,
        type=Path,
        help="Path to the source CSV file.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Path to the output CSV file. Defaults to <input>_feature_parameters_raw.csv.",
    )
    return parser.parse_args()


def resolve_output_path(input_path: Path, output_path: Path | None) -> Path:
    if output_path is not None:
        return output_path
    return input_path.with_name(f"{input_path.stem}_feature_parameters_raw.csv")


def resolve_columns(fieldnames: list[str] | None) -> dict[str, str]:
    if fieldnames is None:
        raise ValueError("Input CSV is missing a header row.")

    mapping: dict[str, str] = {}
    for output_name, aliases in INPUT_ALIASES.items():
        for alias in aliases:
            if alias in fieldnames:
                mapping[output_name] = alias
                break
        else:
            if output_name in OPTIONAL_INPUT_COLUMNS:
                continue
            aliases_text = ", ".join(aliases)
            raise ValueError(f"Required input column not found: {aliases_text}")
    return mapping


def parse_float(row: dict[str, str], column_name: str, row_number: int) -> float:
    raw_value = row[column_name]
    if raw_value is None:
        raise ValueError(f"Row {row_number}: missing column {column_name}")
    value = raw_value.strip()
    if value in PLACEHOLDER_VALUES:
        raise ValueError(
            f"Row {row_number}: missing or placeholder value in column {column_name}"
        )
    try:
        return float(value)
    except ValueError as exc:
        raise ValueError(
            f"Row {row_number}: cannot parse {column_name} as float: {raw_value!r}"
        ) from exc


def parse_source_row(
    row: dict[str, str], column_mapping: dict[str, str], row_number: int
) -> dict[str, float]:
    parsed: dict[str, float] = {}
    errors: list[str] = []
    for output_name, input_name in column_mapping.items():
        try:
            parsed[output_name] = parse_float(row, input_name, row_number)
        except ValueError as exc:
            errors.append(str(exc))

    if errors:
        raise ValueError("; ".join(errors))
    return parsed


def clamp_radius(radius: float, width: float, height: float) -> float:
    if radius <= 0:
        return 0.0
    return min(radius, min(width, height) / 2.0)


def calculate_ix_weak_axis(width: float, height: float, radius: float) -> float:
    if width <= 0 or height <= 0:
        raise ValueError(
            f"Invalid section dimensions for inertia calculation: {width=}, {height=}"
        )

    radius = clamp_radius(radius, width, height)
    if radius == 0:
        return width * height**3 / 12.0

    i1 = width * (height - 2.0 * radius) ** 3 / 12.0

    flange_width = max(width - 2.0 * radius, 0.0)
    flange_height = radius
    flange_area = flange_width * flange_height
    flange_offset = (height - radius) / 2.0
    i2 = 2.0 * (
        flange_width * flange_height**3 / 12.0
        + flange_area * flange_offset**2
    )

    quarter_area = math.pi * radius**2 / 4.0
    quarter_centroid_offset = 4.0 * radius / (3.0 * math.pi)
    quarter_base = math.pi * radius**4 / 16.0
    quarter_centroid = quarter_base - quarter_area * quarter_centroid_offset**2
    quarter_global_offset = (height / 2.0 - radius) + quarter_centroid_offset
    i3 = 4.0 * (
        quarter_centroid + quarter_area * quarter_global_offset**2
    )

    return i1 + i2 + i3


def safe_divide(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0
    return numerator / denominator


def infer_section_family(width: float, height: float, radius: float) -> str:
    aspect_ratio = safe_divide(width, height)
    radius_ratio = safe_divide(radius, height)
    if abs(aspect_ratio - 1.0) <= 1e-6 and abs(radius_ratio - 0.5) <= 1e-3:
        return "circular"
    if abs(aspect_ratio - 1.0) <= 1e-6:
        return "square"
    if radius_ratio > 1e-3:
        return "obround"
    return "rectangular"


def compute_feature_row(source: dict[str, float], row_number: int) -> list[object]:
    b = source["b (mm)"]
    h = source["h (mm)"]
    t = source["t (mm)"]
    r0 = source["r0 (mm)"]
    replacement_ratio = source["R (%)"]
    fy = source["fy (MPa)"]
    fc = source["fc (MPa)"]
    length = source["L (mm)"]
    e1 = source["e1 (mm)"]
    e2 = source["e2 (mm)"]
    nexp = source.get("Nexp (kN)")

    if b <= 0 or h <= 0:
        raise ValueError(f"Row {row_number}: b and h must be positive.")
    if b < h:
        raise ValueError(f"Row {row_number}: expected b >= h, got {b} < {h}.")
    if t < 0:
        raise ValueError(f"Row {row_number}: t must be non-negative.")
    if length <= 0:
        raise ValueError(f"Row {row_number}: L must be positive.")
    if fc <= 0:
        raise ValueError(f"Row {row_number}: fc must be positive.")

    r0 = clamp_radius(r0, b, h)

    inner_width = b - 2.0 * t
    inner_height = h - 2.0 * t
    if inner_width <= 0 or inner_height <= 0:
        raise ValueError(
            f"Row {row_number}: invalid inner dimensions after subtracting wall thickness."
        )

    r1 = clamp_radius(((h - 2.0 * t) / h) * r0, inner_width, inner_height)

    area_concrete = inner_width * inner_height - (4.0 - math.pi) * r1**2
    area_steel = 2.0 * t * (b + h - 4.0 * t) + (4.0 - math.pi) * (r0**2 - r1**2)
    equivalent_radius = math.sqrt(area_concrete / math.pi)
    equivalent_thickness = math.sqrt((area_concrete + area_steel) / math.pi) - equivalent_radius

    flat_width = max(b - 2.0 * r1 - 2.0 * t, 0.0)
    flat_height = max(h - 2.0 * r1 - 2.0 * t, 0.0)
    ke_numerator = (h - 2.0 * t) * flat_width + (b - 2.0 * t) * flat_height
    ke = 1.0 - safe_divide(ke_numerator, 3.0 * area_concrete)

    xi = safe_divide(area_steel * fy, area_concrete * fc)
    sigma_re = ke * safe_divide(equivalent_thickness * fy, equivalent_radius)

    concrete_inertia = calculate_ix_weak_axis(inner_width, inner_height, r1)
    gross_inertia = calculate_ix_weak_axis(b, h, r0)
    steel_inertia = gross_inertia - concrete_inertia
    total_area = area_concrete + area_steel
    total_inertia = concrete_inertia + steel_inertia
    radius_of_gyration = math.sqrt(total_inertia / total_area)
    slenderness = length / radius_of_gyration

    concrete_elastic_modulus = 4700.0 * math.sqrt(fc)
    npl = area_steel * fy + area_concrete * fc
    ncr = (
        math.pi**2
        * (STEEL_ELASTIC_MODULUS * steel_inertia + 0.6 * concrete_elastic_modulus * concrete_inertia)
        / length**2
    )
    lambda_bar = math.sqrt(npl / ncr)

    eccentricity = max(abs(e1), abs(e2))
    eccentricity_ratio = eccentricity / h
    eccentricity_ratio_12 = safe_divide(e1, e2)

    section_modulus = total_inertia / (h / 2.0)
    e_bar = eccentricity * total_area / section_modulus
    axial_flag = "axial" if abs(e_bar) <= 1e-12 else "eccentric"
    section_family = infer_section_family(b, h, r0)
    npl_kn = npl / 1000.0
    psi = safe_divide(nexp, npl_kn) if nexp is not None else None

    return [
        b,
        h,
        r0,
        t,
        replacement_ratio,
        fy,
        fc,
        length,
        e1,
        e2,
        nexp,
        r0 / h,
        safe_divide(b, t),
        area_concrete,
        area_steel,
        equivalent_radius,
        equivalent_thickness,
        ke,
        xi,
        sigma_re,
        slenderness,
        lambda_bar,
        eccentricity_ratio,
        eccentricity_ratio_12,
        e_bar,
        npl_kn,
        psi,
        safe_divide(b, h),
        safe_divide(length, h),
        axial_flag,
        section_family,
    ]


def main() -> None:
    args = parse_args()
    input_path = args.input
    output_path = resolve_output_path(input_path, args.output)

    with input_path.open("r", encoding="utf-8-sig", newline="") as infile:
        reader = csv.DictReader(infile)
        column_mapping = resolve_columns(reader.fieldnames)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8", newline="") as outfile:
            writer = csv.writer(outfile)
            writer.writerow(OUTPUT_COLUMNS)

            written_rows = 0
            skipped_rows: list[str] = []
            for row_number, row in enumerate(reader, start=2):
                try:
                    source = parse_source_row(row, column_mapping, row_number)
                    writer.writerow(compute_feature_row(source, row_number))
                    written_rows += 1
                except ValueError as exc:
                    skipped_rows.append(str(exc))

    print(f"Wrote {written_rows} rows to {output_path}")
    if skipped_rows:
        print(f"Skipped {len(skipped_rows)} rows due to invalid or incomplete data.")
        for detail in skipped_rows[:10]:
            print(f"  - {detail}")


if __name__ == "__main__":
    main()
