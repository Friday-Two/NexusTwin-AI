
import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "datasets/ranking/richer_synthetic_catalog.csv"

ADDITIONS = [
    ("Railway Budget Stay", "Accommodation",
     "Illustrative low-cost lodging; location, prices, and availability unverified."),
    ("Family Airport Lodge", "Accommodation",
     "Illustrative family lodging; airport distance and amenities unverified."),
    ("City Vegetarian Kitchen", "Food",
     "Illustrative vegetarian restaurant; menu, prices, and delivery coverage unverified."),
    ("Healthy Lunch Delivery", "Food",
     "Illustrative lunch delivery; ingredients, nutrition, and delivery area unverified."),
    ("Budget Meal Courier", "Food",
     "Illustrative meal delivery; restaurant partners and delivery times unverified."),
    ("Student Accessories Shop", "Shopping",
     "Illustrative laptop-accessory retailer; inventory and prices unverified."),
    ("Budget Laptop Outlet", "Shopping",
     "Illustrative low-cost computer retailer; stock and prices unverified."),
    ("Computer Parts Store", "Shopping",
     "Illustrative computer-parts shop; compatibility and inventory unverified."),
    ("City Shuttle Service", "Transport",
     "Illustrative city shuttle; routes, schedules, and fares unverified."),
    ("Budget Ride Service", "Transport",
     "Illustrative low-cost ride service; coverage and fares unverified."),
    ("Railway Station Cab", "Transport",
     "Illustrative station transfer service; availability and fares unverified."),
    ("Bus Ticket Planner", "Travel",
     "Illustrative bus-planning service; routes and booking capabilities unverified."),
    ("Rail Journey Assistant", "Travel",
     "Illustrative rail-planning service; schedules and ticket access unverified."),
    ("Intercity Trip Desk", "Travel",
     "Illustrative intercity travel service; destinations and prices unverified."),
    ("Travel Booking Helper", "Travel",
     "Illustrative booking assistance; supported transport modes unverified."),
]

PROVENANCE = "synthetic; attributes are illustrative and unverified"
FIELDS = ["service_name", "category", "description", "catalog_provenance"]

if OUT.exists():
    with OUT.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        existing_fields = reader.fieldnames or []
        if not set(FIELDS).issubset(existing_fields):
            raise ValueError(f"Unexpected catalog columns: {existing_fields}")
        rows = list(reader)
else:
    rows = []

names = {row["service_name"].strip().casefold() for row in rows}
added = 0

for name, category, description in ADDITIONS:
    if name.casefold() in names:
        continue
    rows.append({
        "service_name": name,
        "category": category,
        "description": description,
        "catalog_provenance": PROVENANCE,
    })
    names.add(name.casefold())
    added += 1

with OUT.open("w", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=FIELDS)
    writer.writeheader()
    writer.writerows(rows)

print(f"Updated: {OUT.relative_to(ROOT)}")
print(f"Total rows: {len(rows)}")
print(f"New entries added: {added}")
print(f"Unique service names: {len(names)}")
print("All entries remain explicitly synthetic.")
print("No relevance labels were assigned.")
