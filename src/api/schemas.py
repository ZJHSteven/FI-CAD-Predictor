# -*- coding: utf-8 -*-
# 本文件定义API的请求与响应数据结构（Pydantic模型）。
# 设计目标：
# 1) 用显式字段声明输入特征，便于前端表单生成与校验。
# 2) 将必填与可选字段区分开，允许用户只输入少量关键字段。
# 3) 保留轻量开关，例如是否返回可视化结果。

from typing import Optional, Dict, Any
from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    """
    预测请求结构。

    说明：
    - 必填字段：FI、Age、Gender
    - 其余字段可选；如果缺省，后端会用训练数据的统计值补齐。
    - return_viz 用于控制是否返回可视化结果列表。
    """

    # ===== 必填字段 =====
    FI: float = Field(..., description="虚弱指数（Frailty Index）")
    Age: float = Field(..., description="年龄")
    Gender: float = Field(..., description="性别编码（请与训练数据保持一致）")

    # ===== 可选字段（全部允许为空，后端会自动补默认值） =====
    Marital: Optional[float] = Field(None, description="婚姻状态编码")
    EducationLevel: Optional[float] = Field(None, description="教育水平编码")
    chestPainWhenClimbingOrWalkingQuickly: Optional[float] = Field(None, description="快速行走/爬楼时是否胸痛")
    sleepWasRestless: Optional[float] = Field(None, description="睡眠是否不安稳")
    feelAnyBodyPain: Optional[float] = Field(None, description="是否感到身体疼痛")
    weightChange: Optional[float] = Field(None, description="体重变化情况")
    feelDepressed: Optional[float] = Field(None, description="抑郁程度编码")
    selfCommentOfHealth: Optional[float] = Field(None, description="自评健康状况编码")
    Smoke: Optional[float] = Field(None, description="是否吸烟")
    DrinkAlcohol: Optional[float] = Field(None, description="是否饮酒")

    pulse: Optional[float] = Field(None, description="脉搏")
    whiteBloodCell_inThousands: Optional[float] = Field(None, description="白细胞计数")
    meanCorpuscularVolume: Optional[float] = Field(None, description="平均红细胞体积")
    platelets_10e9perL: Optional[float] = Field(None, description="血小板计数")
    bloodUreaNitrogen_mgperdl: Optional[float] = Field(None, description="尿素氮")
    glucose_mgperdl: Optional[float] = Field(None, description="血糖")
    creatinine_mgperdl: Optional[float] = Field(None, description="肌酐")
    totalCholesterol_mgperdl: Optional[float] = Field(None, description="总胆固醇")
    triglycerides_mgperdl: Optional[float] = Field(None, description="甘油三酯")
    hdlCholesterol_mgperdl: Optional[float] = Field(None, description="高密度脂蛋白胆固醇")
    ldlCholesterol_mgperdl: Optional[float] = Field(None, description="低密度脂蛋白胆固醇")
    cReactiveProtein_mgperdl: Optional[float] = Field(None, description="C反应蛋白")
    glycatedHemoglobin: Optional[float] = Field(None, description="糖化血红蛋白")
    uricAcid_mgperdl: Optional[float] = Field(None, description="尿酸")
    hematocrit: Optional[float] = Field(None, description="红细胞压积")
    hemoglobin_gperdl: Optional[float] = Field(None, description="血红蛋白")
    cystatinC_mgperl: Optional[float] = Field(None, description="胱抑素C")

    # ===== 额外控制字段 =====
    return_viz: bool = Field(True, description="是否返回可视化资源列表")

    class Config:
        # 允许前端传入未知字段（会保留到dict里）
        # 后端会根据配置决定是否允许这些字段继续流入预测
        # 如果你希望严格校验，可将其改为 "forbid"
        extra = "allow"

    def to_feature_dict(self) -> Dict[str, Any]:
        """
        将请求对象转换为“仅特征字段”的字典。

        Returns:
            仅包含模型特征的字典（自动剔除 return_viz）
        """
        data = self.dict()
        # 删除控制字段，确保只保留特征
        data.pop("return_viz", None)
        return data


class PredictResponse(BaseModel):
    """
    预测响应结构。

    说明：
    - label: 最终分类结果（0/1）
    - probability: 正类概率（风险分数）
    - model_details: 各子模型的输出（便于调试或展示）
    - used_models: 实际参与集成的模型列表
    - figures: 可选的可视化资源URL列表
    """

    label: int = Field(..., description="最终分类标签（0/1）")
    probability: float = Field(..., description="正类概率（0~1）")
    model_details: Dict[str, float] = Field(..., description="各子模型的预测概率")
    used_models: Dict[str, float] = Field(..., description="各子模型的权重")
    figures: Optional[list] = Field(None, description="可视化资源URL列表")
