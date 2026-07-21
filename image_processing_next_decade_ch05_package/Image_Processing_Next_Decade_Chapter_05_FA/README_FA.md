# فصل پنجم: بهبود تصویر، حذف نویز و بازسازی

این بسته، فصل پنجم کتاب/جزوه «پردازش تصویر در دهه آینده: از عمق ریاضیات تا مهندس حرفه‌ای» است.

## محتوای بسته

- `image_processing_next_decade_ch05_fa.tex`: متن کامل فارسی و ریاضی‌محور فصل.
- `image_processing_next_decade_ch05_fa.pdf`: نسخه کامپایل‌شده با XeLaTeX.
- `figures/`: شکل‌های PDF و PNG قابل بازتولید.
- `scripts/generate_figures.py`: تولید شکل‌ها و آزمایش‌های عددی فصل.
- `outputs/`: نتایج CSV تحلیل حساسیت و مقایسه روش‌ها.
- `requirements.txt`: وابستگی‌های پایتون.

## کامپایل LaTeX

در TeXstudio موتور پیش‌فرض را روی XeLaTeX قرار دهید و فایل را دو بار کامپایل کنید:

```bash
xelatex image_processing_next_decade_ch05_fa.tex
xelatex image_processing_next_decade_ch05_fa.tex
```

فونت اصلی `Amiri`، فونت لاتین `DejaVu Serif` و فونت کد `DejaVu Sans Mono` است. برای نبودن فونت‌ها جایگزین خودکار در فایل تعریف شده و هیچ فایل فونتی در بسته توزیع نشده است.

## بازتولید شکل‌ها

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
python scripts/generate_figures.py
```

اسکریپت از تصاویر نمونه داخلی `scikit-image` استفاده می‌کند و به اینترنت وابسته نیست. بذر تصادفی ثابت است.

## کنترل انجام‌شده

- سه بار کامپایل با XeLaTeX
- کنترل نبودن `Missing character`
- تعبیه فونت‌های فارسی، لاتین و کد در PDF
- رندر همه صفحات و بررسی نمونه‌های پراکنده از عنوان تا منابع
- ثبت نتایج عددی در CSV
