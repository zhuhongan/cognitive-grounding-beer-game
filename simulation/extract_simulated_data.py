import argparse
import json
import os
from glob import glob
from typing import Any, Dict, List, Tuple


ROLE_KEY_TO_PREFIX: Dict[str, str] = {
    "Retailer": "retailer",
    "Wholesaler": "wholesaler",
    "Distributor": "distributor",
    "Factory": "factory",
}


def _reconstruct_from_history(simulation_results: Dict[str, Any]) -> Tuple[Dict[str, List[int]], Dict[str, List[int]]]:
    """Fallback reconstruction if top-level 'orders' or 'inventories' are missing.

    Rebuild dicts of role->orders and role->inventories using 'history' entries
    with fields 'order_placed' and 'inventory'.
    """
    orders_by_role: Dict[str, List[int]] = {}
    inventories_by_role: Dict[str, List[int]] = {}

    history: Dict[str, Any] = simulation_results.get("history", {})
    for role, short in ROLE_KEY_TO_PREFIX.items():
        entries: List[Dict[str, Any]] = history.get(role, [])
        orders_by_role[role] = [int(e.get("order_placed", 0)) for e in entries]
        inventories_by_role[role] = [int(e.get("inventory", 0)) for e in entries]

    return orders_by_role, inventories_by_role


def _reconstruct_backlogs_from_history(simulation_results: Dict[str, Any]) -> Dict[str, List[int]]:
    """Reconstruct backlogs from history entries when top-level 'backlogs' is missing.

    For the new format, history entries have an explicit 'backlog' field.
    """
    backlogs_by_role: Dict[str, List[int]] = {}

    history: Dict[str, Any] = simulation_results.get("history", {})
    for role in ROLE_KEY_TO_PREFIX:
        entries: List[Dict[str, Any]] = history.get(role, [])
        backlogs_by_role[role] = [int(e.get("backlog", 0)) for e in entries]

    return backlogs_by_role


def _is_new_format(simulation_results: Dict[str, Any]) -> bool:
    """Detect whether a simulation result uses the new fixed format.

    The new format has explicit top-level 'backlogs' and/or 'shipments' keys,
    and inventories are always non-negative (backlog is tracked separately).
    """
    return "backlogs" in simulation_results or "shipments" in simulation_results


def extract_simulated_data_from_file(file_path: str) -> Tuple[Dict[str, Any], List[str]]:
    """Extract required arrays from Beer Game result JSONs.

    Supports both the old format (negative inventories encode backlog) and the
    new fixed format (non-negative inventories with explicit backlogs, shipments,
    cost_reasoning, and richer financial_results).

    Returns (extracted_data, missing_fields_info) where missing_fields_info contains
    high-level missing structures encountered (e.g., 'orders', 'inventories').
    """
    with open(file_path, "r") as file:
        simulation_results: Dict[str, Any] = json.load(file)

    missing: List[str] = []
    new_format: bool = _is_new_format(simulation_results)

    # ---- Orders ----
    orders_by_role: Dict[str, List[int]] = simulation_results.get("orders") or {}

    # ---- Inventories ----
    inventories_by_role: Dict[str, List[int]] = simulation_results.get("inventories") or {}

    # ---- Backlogs (new format only) ----
    backlogs_by_role: Dict[str, List[int]] = simulation_results.get("backlogs") or {}

    # ---- Shipments (new format only) ----
    shipments_by_role: Dict[str, List[int]] = simulation_results.get("shipments") or {}

    # Fallback to reconstruct from 'history' if needed
    if not orders_by_role or not inventories_by_role:
        rec_orders, rec_inventories = _reconstruct_from_history(simulation_results)
        if not orders_by_role:
            orders_by_role = rec_orders
            missing.append("orders (reconstructed from history)")
        if not inventories_by_role:
            inventories_by_role = rec_inventories
            missing.append("inventories (reconstructed from history)")

    if new_format and not backlogs_by_role:
        backlogs_by_role = _reconstruct_backlogs_from_history(simulation_results)
        missing.append("backlogs (reconstructed from history)")

    extracted: Dict[str, Any] = {}

    for role, prefix in ROLE_KEY_TO_PREFIX.items():
        role_orders: List[int] = [int(v) for v in list(orders_by_role.get(role, []))]

        if new_format:
            # New format: inventories are already non-negative, backlogs are explicit.
            role_inventories_clean: List[int] = [int(v) for v in list(inventories_by_role.get(role, []))]
            role_backlog: List[int] = [int(v) for v in list(backlogs_by_role.get(role, []))]
        else:
            # Old format: inventories can be negative; negative values represent backlog.
            role_inventories_raw: List[int] = [int(v) for v in list(inventories_by_role.get(role, []))]
            role_backlog = [max(-v, 0) for v in role_inventories_raw]
            role_inventories_clean = [max(v, 0) for v in role_inventories_raw]

        extracted[f"{prefix}_order"] = role_orders
        extracted[f"{prefix}_inventory"] = role_inventories_clean
        extracted[f"{prefix}_backlog"] = role_backlog

        # Extract shipments if available (new format)
        if role in shipments_by_role:
            role_shipments: List[int] = [int(v) for v in list(shipments_by_role[role])]
            extracted[f"{prefix}_shipment"] = role_shipments

        # Track missing per-role arrays at a high level
        if role not in orders_by_role:
            missing.append(f"orders[{role}]")
        if role not in inventories_by_role:
            missing.append(f"inventories[{role}]")
        if new_format and role not in backlogs_by_role:
            missing.append(f"backlogs[{role}]")

    # Extract financial_results if available (new format)
    financial_results: Dict[str, Any] = simulation_results.get("financial_results") or {}
    if financial_results:
        extracted["financial_results"] = {}
        for role, prefix in ROLE_KEY_TO_PREFIX.items():
            if role in financial_results:
                extracted["financial_results"][prefix] = financial_results[role]

    return extracted, missing


def write_json(output_path: str, payload: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as file:
        json.dump(payload, file, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Extract specific simulated data arrays from Beer Game simulation results JSONs."
        )
    )
    parser.add_argument(
        "--input-dir",
        default="oss20b-prompt-06",
        help="Directory containing sim_*.json files (default: fewshot_sims)",
    )
    parser.add_argument(
        "--output-dir",
        default="oss20b-prompt-06/oss20b-prompt-06",
        help=(
            "Directory to write per-file extracted JSONs. Defaults to <input-dir>/extracted"
        ),
    )
    parser.add_argument(
        "--aggregate-path",
        default=None,
        help=(
            "Optional path to also write a single aggregated JSON mapping filename to extracted data."
        ),
    )

    args = parser.parse_args()

    input_dir: str = os.path.abspath(args.input_dir)
    output_dir: str = (
        os.path.abspath(args.output_dir)
        if args.output_dir
        else os.path.join(input_dir, "extracted")
    )

    json_paths: List[str] = sorted(glob(os.path.join(input_dir, "sim_*.json")))
    if not json_paths:
        print(
            f"No files matched pattern 'sim_*.json' in directory: {input_dir}. Nothing to do."
        )
        return

    os.makedirs(output_dir, exist_ok=True)

    aggregated: Dict[str, Dict[str, Any]] = {}
    total_files: int = 0
    files_with_missing: int = 0

    for file_path in json_paths:
        total_files += 1
        base_name = os.path.splitext(os.path.basename(file_path))[0]

        try:
            extracted, missing = extract_simulated_data_from_file(file_path)
        except Exception as exc:  # noqa: BLE001
            print(f"ERROR reading {file_path}: {exc}")
            continue

        if missing:
            files_with_missing += 1
            print(
                f"WARNING {base_name}: missing keys in 'data': {', '.join(missing)}"
            )

        # Save per-file extracted JSON
        output_path = os.path.join(output_dir, f"{base_name}_extracted.json")
        write_json(output_path, extracted)

        # Add to aggregated mapping
        aggregated[base_name] = extracted

    # Optionally write aggregated file
    if args.aggregate_path:
        aggregate_path = os.path.abspath(args.aggregate_path)
        write_json(aggregate_path, aggregated)
        print(f"Aggregated extracted data written to: {aggregate_path}")

    print(
        f"Processed {total_files} file(s). Per-file outputs at: {output_dir}. Files with missing keys: {files_with_missing}."
    )


if __name__ == "__main__":
    main()
