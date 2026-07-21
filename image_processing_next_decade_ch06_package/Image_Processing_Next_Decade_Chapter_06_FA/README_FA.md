# فصل ششم: هندسه تصویر و دوربین

این بسته شامل فایل LaTeX، PDF نهایی، شکل‌های بازتولیدپذیر، کد پایتون و خروجی‌های عددی فصل ششم کتاب «پردازش تصویر در دهه آینده» است.

## کامپایل

1. TeX Live و TeXstudio را نصب کنید.
2. کامپایلر را روی XeLaTeX قرار دهید.
3. فایل `image_processing_next_decade_ch06_fa.tex` را دو بار کامپایل کنید.

## تولید شکل‌ها

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux/macOS
source .venv/bin/activate
pip install -r requirements.txt
python scripts/generate_figures.py
```

## فونت‌ها

فایل از Amiri، DejaVu Serif و DejaVu Sans Mono استفاده می‌کند و در صورت نبودن فونت‌ها جایگزین خودکار دارد. هیچ فایل فونتی در بسته توزیع نشده است.

## خروجی‌های عددی

- `outputs/calibration_summary.csv`
- `outputs/calibration_per_view.csv`
- `outputs/estimated_fundamental_matrix.csv`
- `outputs/epipolar_metrics.csv`
