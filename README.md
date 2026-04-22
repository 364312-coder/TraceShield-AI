# TraceShield-AI

## 项目简介

TraceShield 是一个用于 AI 模型版权保护的系统，通过在模型中嵌入水印，并通过行为触发进行检测，实现模型归属验证。

---

## 功能

* 模型训练（train.py）
* 水印嵌入（train_watermark.py）
* 水印检测（detect.py）

---

## 环境配置

```bash
conda create -n TraceShield python=3.9
conda activate TraceShield
pip install -r requirements.txt
```

---

## 运行方法

```bash
python detect.py
```

---

## 模型下载

由于模型文件较大（>25MB），未上传至 GitHub。

请将以下文件放入项目根目录：

* clean_model.pth
* watermark_model.pth

（模型由项目作者单独提供）

---

## 实验结果

* clean_model：无水印（触发率≈0.08）
* watermark_model：有水印（触发率≈0.88）

---

## 项目说明

本项目通过设计触发器，使带水印模型对特定输入产生固定响应，从而实现可验证的模型水印检测。
