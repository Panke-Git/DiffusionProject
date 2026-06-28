```json
{
  "V1": {
    "name": "Baseline_UAS4",
    "note": "基线+UAS4",
    "相关文件": "NULL"
  },
  "V3": {
    "name": "UNet_UAS4_ADMM_V3",
    "note": "UAS4+深度自适应正则化",
    "相关文件": "[depth_estimator_admm.py][depth_guided_adaptive_regularization.py]"
  },
  "V4": {
    "name": "UNet_ADMM_V4",
    "note": "深度自适应正则化+基础Unet",
    "相关文件": "[depth_estimator_admm.py][depth_guided_adaptive_regularization.py]"
  },
  "V6": {
    "name": "V6_UAS4_DGRM_A4",
    "note": "基于 V1/Baseline_UAS4，在 decoder 中加入 DGRM-A4 深度引导双频重建调制",
    "主要内容": [
      "保留 V1 的 Baseline_UAS4 主体结构和 mid_uas 模块",
      "使用 SceneDepthEstimatorADMM 从条件输入估计伪深度图，训练和采样时统一传入 UNet",
      "在 decoder 指定分辨率的 ResBlock 后插入 DGRM_A4，默认配置为 64 尺度 stage_last",
      "DGRM_A4 将 decoder 特征分解为低频结构和高频细节，并由 depth_g/depth_edge 生成 gate、gamma、beta 调制",
      "low_out 和 high_out 采用零初始化，并在全局初始化后重置，保证初始阶段 F_l' 近似 F_l"
    ],
    "相关文件": "[model/DGRM_A4.py][model/ddpm_modules/unetV6.py][model/ddpm_modules/diffusionV6.py][model/networksV6.py][model/modelV6.py][config/config6.yaml][train6.py]"
  },
  "V7": {
    "name": "V7_UAS4_ADMM_DGRMPlus",
    "note": "基于 V3/UNet_UAS4_ADMM_V3，加入 decoder 侧 DGRMPlus 深度引导重建调制",
    "主要内容": [
      "保留 V3 的 UAS4 主体结构和 ADMM 深度自适应正则训练损失",
      "沿用 SceneDepthEstimatorADMM 从条件输入估计伪深度图",
      "训练时同一张 depth_map 同时传入 UNet 的 DGRMPlus，并用于 DepthGuidedAdaptiveRegularizer",
      "采样时从条件输入估计 condition_depth，并贯穿每一步反向扩散传入 UNet",
      "DGRMPlus 默认在 decoder 的 32、64、128、256 分辨率 stage_last 位置插入"
    ],
    "相关文件": "[model/DGRM_PLUS.py][model/ddpm_modules/unetV7.py][model/ddpm_modules/diffusionV7.py][model/networksV7.py][model/modelV7.py][config/config7.yaml][train7.py]"
  }
}
```
