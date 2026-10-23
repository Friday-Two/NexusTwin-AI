import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from sqlalchemy import select

from database.connection import SessionLocal
from database.models.entities import Service

SAMPLE_SERVICES = [
    ("Budget Hotel", "Accommodation",
     "Affordable hotel accommodation for budget-conscious travelers."),
    ("Premium City Hotel", "Accommodation",
     "Comfortable city hotel with premium amenities."),
    ("Airport Taxi", "Transport",
     "Convenient airport pickup and drop-off transport."),
    ("Train Ticket Assistance", "Travel",
     "Travel planning and train ticket assistance."),
    ("Student Laptop Store", "Shopping",
     "Affordable laptops and accessories for students."),
    ("Local Food Delivery", "Food",
     "Local restaurant meals delivered to your location."),
]

def main():
    added = 0

    with SessionLocal() as db:
        try:
            for name, category, description in SAMPLE_SERVICES:
                existing = db.scalar(
                    select(Service).where(
                        Service.name == name,
                        Service.category == category,
                    )
                )

                if existing is None:
                    db.add(Service(
                        name=name,
                        category=category,
                        description=description,
                        active=True,
                    ))
                    added += 1

            db.commit()

            services = db.scalars(
                select(Service)
                .where(Service.active.is_(True))
                .order_by(Service.name)
            ).all()

            print(f"Sample services added: {added}")
            print(f"Total active services: {len(services)}")

            for service in services:
                print(f"{service.id}. {service.name} | {service.category}")

        except Exception:
            db.rollback()
            raise

if __name__ == "__main__":
    main()
