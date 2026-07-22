# فصل هشتم: قطعه‌بندی تصویر و استخراج ناحیه

این بسته شامل متن کامل LaTeX، PDF کامپایل‌شده، شکل‌های بازتولیدپذیر، کد تولید شکل‌ها و یک آزمایش مستقل قطعه‌بندی است.

## کامپایل متن

موتور لازم: XeLaTeX. فونت‌های پیشنهادی: Amiri، DejaVu Serif و DejaVu Sans Mono.

```bash
xelatex image_processing_next_decade_ch08_fa.tex
xelatex image_processing_next_decade_ch08_fa.tex
```

## تولید دوباره شکل‌های فصل

```bash
python generate_figures.py
```

شکل‌ها در پوشه `figures` و جدول نمونه در `outputs` نوشته می‌شوند.

## اجرای آزمایش مستقل

```bash
python -m venv .venv
# در ویندوز: .venv\\Scripts\\activate
# در لینوکس/macOS: source .venv/bin/activate
pip install -r requirements.txt
python chapter08_experiments.py --output chapter08_results --seed 1403
```

آزمایش مستقل، Otsu، Sauvola، Random Walker و Chan--Vese را روی یک صحنه کنترل‌شده مقایسه می‌کند، اثر پس‌پردازش مورفولوژیک را می‌سنجد و معیارهای IoU، Dice، Precision، Recall و Balanced Accuracy را در CSV می‌نویسد. داده مصنوعی فقط برای کنترل صحت و آموزش است و جایگزین ارزیابی روی داده واقعی نیست.

## نکات بازتولیدپذیری

- seed همه آزمایش‌های تصادفی ثابت است.
- خروجی نسخه کتابخانه‌ها در `metadata.json` ثبت می‌شود.
- هیچ داده‌ای از اینترنت هنگام اجرا دریافت نمی‌شود.
- فایل‌های فونت در این بسته توزیع نشده‌اند.
