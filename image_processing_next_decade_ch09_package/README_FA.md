# فصل نهم: ویژگی، تطبیق و بازشناسی کلاسیک

## ساخت PDF

```bash
xelatex image_processing_next_decade_ch09_fa.tex
xelatex image_processing_next_decade_ch09_fa.tex
```

## بازتولید شکل‌ها و آزمایش‌ها

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
python generate_figures.py
python chapter09_experiments.py --output outputs
```

خروجی آزمایش‌ها آموزشی است و جای benchmark رسمی را نمی‌گیرد. برای پژوهش، split گروهی، داده واقعی، فاصله اطمینان و ثبت کامل محیط اجرا لازم است.

فونت پیشنهادی فارسی `Amiri` است و در صورت نبودن آن `Noto Naskh Arabic` انتخاب می‌شود. فایل فونت در بسته قرار ندارد.
