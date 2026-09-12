"""
config/yaml_config.py
========================
負責讀取專案根目錄的 config.yaml，取得各個模型
（Embedding / LLM / VLM）目前選用的 provider 與 model 名稱。

為什麼另外用 YAML，而不是全部塞進 settings.py（.env）？
- .env / Settings（settings.py）：機密設定（API Key）與部署環境設定
  （資料庫路徑、DEBUG 開關），因環境而異，不應該進版本控制。
- config.yaml：「要用哪個模型」這種需要團隊協作、常態調整、
  應該被 git 追蹤變更歷史的設定，修改它不需要碰任何程式碼。

embedding / visual_embedding / llm / vlm 四個區塊都已經串到對應的 factory
（見 embedding/factory.py、llm/factory.py、vlm/factory.py；
visual_embedding 跟 embedding 共用 EmbeddingFactory，只是 provider/model
不同），app/dependencies.py 的 get_embedding_client() /
get_visual_embedding_client() / get_llm_client() / get_vlm_client()
都會讀取這裡的設定動態建立實作。
"""

from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Optional

import yaml
from pydantic import BaseModel

# config.yaml 放在 project/ 根目錄（跟 requirements.txt 同層）
_DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent / "config.yaml"


class ProviderConfig(BaseModel):
    """單一模型類別（embedding / llm / vlm）的 provider + model 設定。"""

    provider: str
    model: str
    extra: Dict[str, Any] = {}


class AppConfig(BaseModel):
    """config.yaml 的完整結構。"""

    embedding: ProviderConfig
    visual_embedding: ProviderConfig
    llm: ProviderConfig
    vlm: ProviderConfig


@lru_cache
def load_config(config_path: Optional[Path] = None) -> AppConfig:
    """
    讀取並解析 config.yaml（單例快取，同一個路徑只會讀取/解析一次）。

    Args:
        config_path: 自訂設定檔路徑，預設讀取 project 根目錄的 config.yaml。
    """
    path = config_path or _DEFAULT_CONFIG_PATH
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    return AppConfig(**raw)
