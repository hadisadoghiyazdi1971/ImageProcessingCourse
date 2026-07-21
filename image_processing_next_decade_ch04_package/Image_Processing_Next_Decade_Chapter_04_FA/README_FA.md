# فصل چهارم: تصویر تصادفی و مدل‌های نویز

این بسته فصل چهارم کتاب/درس **پردازش تصویر در دهه آینده: از عمق ریاضیات تا مهندس حرفه‌ای** است.
قالب، رنگ‌بندی، محیط‌های آموزشی و تنظیمات فارسی با فصل‌های دوم و سوم هماهنگ شده‌اند.

## محتوای علمی

- تصویر به‌عنوان بردار و میدان تصادفی
- ایستایی، ارگودیسیته، همسانگردی، سفیدی و وابستگی فضایی
- قضیه وینر--خینچین، PSD، دوره‌نگار و واریوگرام
- فیزیک نویز حسگر: shot/read/dark/DSNU/PRNU/quantization
- مدل‌های Gaussian، Poisson، Poisson--Gaussian، impulse، speckle و correlated
- MLE، Fisher information، CRLB و MAP
- dark frame، flat field و photon-transfer curve
- residual diagnostics و model checking
- Anscombe و generalized variance stabilization
- Noise2Noise، Noise2Void و Noise2Self با استخراج ریاضی فرض‌ها
- کد بازتولیدپذیر، تمرین و تقسیم ۱۶ ویدئوی ۱۵ دقیقه‌ای

## ساخت شکل‌ها

```powershell
py scripts/generate_figures.py
```

یا در Linux/macOS:

```bash
python3 scripts/generate_figures.py
```

## ساخت PDF

در TeXstudio کامپایلر را روی XeLaTeX بگذارید، یا اجرا کنید:

```bash
xelatex image_processing_next_decade_ch04_fa.tex
xelatex image_processing_next_decade_ch04_fa.tex
```

## فونت

فونت اصلی `Amiri` است و در صورت نبودن به `Noto Naskh Arabic` برمی‌گردد. فونت لاتین `DejaVu Serif`
و فونت کد `DejaVu Sans Mono` است. هیچ فایل فونتی در بسته قرار نگرفته است.
