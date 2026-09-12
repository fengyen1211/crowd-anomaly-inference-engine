"""
config package
=================
- settings.py     Settings（.env）：機密設定與部署環境設定
                  （API Key、資料庫路徑、DEBUG 開關）
- yaml_config.py  AppConfig（config.yaml）：目前選用的 Embedding / LLM / VLM
                  provider 與 model 名稱，這是需要被版本控制追蹤、
                  常態調整的設定，跟 settings.py 的性質不同。

之後若要更換 Embedding、VLM 或 LLM 供應商，主要修改點都是專案根目錄的
config.yaml（三者都已經改用 factory pattern，見各自 package 的
factory.py），不需要改動 settings.py 或任何業務邏輯程式碼。
"""
