"""
vlm package
=============
負責「視覺語言模型（VLM）」相關功能。

★ 架構跟 embedding/ 套件完全對稱 ★

- base.py         BaseVLM 抽象介面（load_model / describe_image / generate_embedding）
- factory.py      VLMFactory：依 provider 名稱建立實例
- mock.py         MockVLM：★ 目前唯一可用的 Provider ★，不載入任何真實模型
- qwen25_vl.py    Qwen2.5-VL：骨架，之後才實作
- internvl.py     InternVL：骨架，之後才實作
- llava.py        LLaVA：骨架，之後才實作
- minicpm_v.py    MiniCPM-V：骨架，之後才實作
- schemas.py      VLMObservation：業務層輸出結構，跟 Provider 抽象層屬於不同層級
- observation_builder.py  VLMObservationBuilder：把 Provider 的
                  describe_image() 原始輸出組成 VLMObservation 的業務層
"""
