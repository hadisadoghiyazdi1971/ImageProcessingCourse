# فصل یازدهم: ترنسفورمرهای بینایی و یادگیری خودنظارتی

این بسته، فصل یازدهم کتاب «پردازش تصویر در دهه آینده: از عمق ریاضیات تا مهندس حرفه‌ای» را در بر دارد.

## اجزای بسته

- `image_processing_next_decade_ch11_fa.tex`: متن کامل فارسی LaTeX
- `image_processing_next_decade_ch11_fa.pdf`: نسخه کامپایل‌شده ۸۵ صفحه‌ای
- `figures/`: ۲۴ شکل علمی در دو قالب PDF و PNG
- `code/`: نه برنامه مستقل پایتون
- `outputs/`: نتایج آزمون‌های دود، جدول هزینه توکن و فهرست شکل‌ها
- `generate_figures.py`: تولید بازتولیدپذیر شکل‌های فصل
- `requirements.txt`: وابستگی‌های پیشنهادی
- `manifest_sha256.txt`: کنترل تمامیت فایل‌های اصلی بسته

## ساخت PDF

در TeXstudio، کامپایلر را روی XeLaTeX قرار دهید و فایل اصلی را دو بار کامپایل کنید. در خط فرمان:

```bash
xelatex -interaction=nonstopmode image_processing_next_decade_ch11_fa.tex
xelatex -interaction=nonstopmode image_processing_next_decade_ch11_fa.tex
```

فونت‌های استفاده‌شده عبارت‌اند از Amiri، DejaVu Serif و DejaVu Sans Mono. فایل فونت در بسته قرار نگرفته است.

## تولید شکل‌ها

```bash
python generate_figures.py
```

## اجرای کدهای آموزشی

```bash
python code/patchify.py
python code/minimal_vit.py
python code/info_nce.py
python code/simclr_minimal.py
python code/mae_minimal.py
python code/dino_ema.py
python code/native_resolution_pack.py
python code/benchmark_attention.py --tokens 64
```

فایل `inspect_pretrained.py` برای بررسی مدل‌های پیش‌آموخته به نصب `timm` نیاز دارد:

```bash
pip install timm
python code/inspect_pretrained.py
```

کدها آموزشی‌اند و برای آموزش وب‌مقیاس باید data loader توزیع‌شده، checkpoint مقاوم، ثبت آزمایش، مدیریت mixed precision و کنترل تکرارپذیری به آن‌ها افزوده شود.
