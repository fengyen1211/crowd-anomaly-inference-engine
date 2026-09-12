"""
llm package
=============
負責「大型語言模型（LLM）推理」相關功能：

- base.py               BaseLLMClient：LLM 客戶端抽象介面（支援未來更換供應商）
- factory.py            LLMFactory：依 provider 名稱建立實例
- mock.py               MockLLMClient：★ 目前預設 Provider ★，不需要任何外部依賴
- openai_llm.py         OpenAILLMClient：可選 Provider（透過 LangChain 呼叫，需 API Key）
- reasoning_pipeline.py ReasoningPipeline：整合 VLM 觀察 + RAG 檢索結果，
                        透過 LLM 呼叫產生最終分析結果
- schemas.py            AnalysisResult：Reasoning Module 的輸出資料結構
"""
