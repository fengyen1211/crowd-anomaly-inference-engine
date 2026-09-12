"""
vlm/factory.py
=================
VLM Provider 的工廠。

依照傳入的 provider_name，動態建立對應的 BaseVLM 實例。
跟 embedding/factory.py 完全對稱的設計：在每個分支裡才 import 對應的
Provider 類別，避免要求使用者一次裝齊所有模型的相依套件——每個本地
VLM 底層都需要 transformers/torch，權重甚至可能上看數十 GB，
用哪個就只裝哪個。

"mock" 跟 "ollama" 是真的可以用的 Provider；qwen2.5-vl / internvl /
llava / minicpm_v 這幾個 transformers 版本的骨架則保留給之後若不走
Ollama、需要更細緻控制（例如自訂 generate_embedding）時再實作，
目前這些候選模型都直接透過 "ollama" provider 呼叫（見 vlm/ollama_vlm.py），
實際呼叫骨架版本時會丟出 NotImplementedError，明確提示「還沒接上真實模型」，
而不是靜默失敗或退回假資料。
"""

from typing import Optional

from vlm.base import BaseVLM

# 支援的 provider 名稱，對應 config.yaml 的 vlm.provider 欄位
SUPPORTED_PROVIDERS = ("mock", "qwen2.5-vl", "internvl", "llava", "minicpm_v", "ollama")


class VLMFactory:
    """依名稱建立 VLM Provider 的工廠類別。"""

    @classmethod
    def create(
        cls,
        provider_name: str,
        model: Optional[str] = None,
        **kwargs,
    ) -> BaseVLM:
        """
        Args:
            provider_name: "mock" / "qwen2.5-vl" / "internvl" / "llava" / "minicpm_v"
            model: 模型名稱或權重路徑，不同 provider 有各自的預設值，可不傳。
            **kwargs: 傳給對應 Provider 建構子的其他參數（例如 device）。

        Raises:
            ValueError: provider_name 不在支援清單內。
        """
        if provider_name == "mock":
            from vlm.mock import MockVLM

            return MockVLM(**kwargs)

        if provider_name == "qwen2.5-vl":
            from vlm.qwen25_vl import Qwen25VLProvider

            return Qwen25VLProvider(model_name=model or "Qwen/Qwen2.5-VL-7B-Instruct", **kwargs)

        if provider_name == "internvl":
            from vlm.internvl import InternVLProvider

            return InternVLProvider(model_name=model or "OpenGVLab/InternVL2-8B", **kwargs)

        if provider_name == "llava":
            from vlm.llava import LLaVAProvider

            return LLaVAProvider(model_name=model or "llava-hf/llava-1.5-7b-hf", **kwargs)

        if provider_name == "minicpm_v":
            from vlm.minicpm_v import MiniCPMVProvider

            return MiniCPMVProvider(model_name=model or "openbmb/MiniCPM-V-2_6", **kwargs)

        if provider_name == "ollama":
            from vlm.ollama_vlm import OllamaVLM

            return OllamaVLM(model=model or "qwen2.5vl:3b", **kwargs)

        raise ValueError(
            f"不支援的 VLM provider：'{provider_name}'，"
            f"支援的選項：{SUPPORTED_PROVIDERS}"
        )
