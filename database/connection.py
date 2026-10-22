
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from configs.settings import PROJECT_ROOT, get_settings
from database.models.entities import Base

settings = get_settings()
DATABASE_URL = settings.database_url

connect_args = {}
if DATABASE_URL.startswith("sqlite"):
    connect_args["check_same_thread"] = False

if DATABASE_URL.startswith("sqlite:///") and not DATABASE_URL.startswith("sqlite:////"):
    relative_path = DATABASE_URL.removeprefix("sqlite:///").split("?", 1)[0]
    database_path = (PROJECT_ROOT / relative_path).resolve()
    database_path.parent.mkdir(parents=True, exist_ok=True)
    DATABASE_URL = "sqlite:///" + database_path.as_posix()

engine = create_engine(
    DATABASE_URL,
    connect_args=connect_args,
)

SessionLocal = sessionmaker(
    bind=engine,
    autoflush=False,
    expire_on_commit=False,
)


def init_db() -> None:
    Base.metadata.create_all(bind=engine)
