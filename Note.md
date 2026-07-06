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
  },
  "V8": {
    "name": "V8_Baseline_ADMM_DepthAdaptiveReg",
    "note": "Baseline + ADMM 深度自适应正则 loss",
    "主要内容": [
      "使用原始 baseline UNet，不加入 UAS4、DGRM_A4 或 DGRMPlus 结构改动",
      "训练时从条件输入通过 SceneDepthEstimatorADMM 估计伪深度图",
      "diffusion loss 使用按像素归一化的噪声预测损失，便于和正则项保持相近量级",
      "从 noise_pred 反推 x0_pred，并在 [0,1] 空间计算 DepthGuidedAdaptiveRegularizer",
      "总损失为 diffusion_loss + lambda_reg * depth_adaptive_reg_loss",
      "日志记录 l_pix、l_depth_reg、l_total、l_adaptive_tv、l_edge_align 和 lambda_depth_reg"
    ],
    "相关文件": "[model/depth_estimator_admm.py][model/depth_guided_adaptive_regularization.py][model/ddpm_modules/diffusionV8.py][model/networksV8.py][model/modelV8.py][config/config8.yaml][train8.py][run_train8.sh]"
  },
  "V10": {
    "name": "V10_Baseline_ADMM_DLDG",
    "note": "基于 Baseline，加入 DepthGuidedLuminanceGate 深度引导亮度门控",
    "主要内容": [
      "保留原始 Baseline UNet 结构，不改 decoder/encoder 主干",
      "沿用 SceneDepthEstimatorADMM 从条件输入估计伪深度图，供 DLDG 使用",
      "DLDG 在 Lab 空间只调制预测 x0 的 L 通道，保留主分支输出的 a/b 色度",
      "训练时先由 Baseline 噪声预测反推出 x0_pred，再经过 DLDG 得到 x0_gated，并换算为等效噪声计算 diffusion loss",
      "采样时每一步反向扩散都在 posterior 前对 x_recon 施加 DLDG",
      "由于数据集输入/GT 使用 [-1,1]，DLDG 默认配置 input_range/output_range 均为 -1_1"
    ],
    "相关文件": "[model/depth_guided_luminance_gate.py][model/ddpm_modules/diffusionV10.py][model/networksV10.py][model/modelV10.py][config/config10.yaml][train10.py][run_train10.sh]"
  }
}
```
