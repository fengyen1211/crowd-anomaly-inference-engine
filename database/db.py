"""
database/db.py
================
資料庫連線管理（SQLAlchemy）。

負責：
- 建立資料庫 engine
- 提供 Session 給各個 service 使用（透過 FastAPI 的 Depends）

TODO：
- [ ] 依 config.settings.database_url 建立 engine：
      sqlalchemy.create_engine(self._settings.database_url)
- [ ] 建立 sessionmaker：
      sqlalchemy.orm.sessionmaker(bind=self._engine)
- [ ] 實作 get_session() / get_db()，給 FastAPI Depends() 使用
      （建議用 contextmanager 或 yield 模式，確保正確關閉連線）
- [ ] 加入 Alembic migration 設定（目前先用 create_all 應急）
"""

from config.settings import Settings, get_settings
from utils.logger import get_logger

logger = get_logger(__name__)


class DatabaseManager:
    """
    集中管理 SQLAlchemy engine 與 session 的建立。
    """

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()
        self._engine = None           # TODO: sqlalchemy.create_engine(...)
        self._session_factory = None  # TODO: sqlalchemy.orm.sessionmaker(bind=self._engine)

    def init_db(self) -> None:
        """
        建立所有資料表。

        TODO: 呼叫 database.models.Base.metadata.create_all(self._engine)
              （正式環境應改用 Alembic migration，而不是 create_all）。
        """
        logger.info("DatabaseManager.init_db() 尚未實作")
        raise NotImplementedError

    def get_session(self):
        """
        取得一個新的資料庫 Session。

        TODO: 回傳一個新的 SQLAlchemy Session，
              並確保使用完畢後正確關閉。
        """
        logger.info("DatabaseManager.get_session() 尚未實作")
        raise NotImplementedError
