# فصل دوازدهم: مدل‌های مولد تصویر

این بسته بخشی از کتاب/جزوه «پردازش تصویر در دهه آینده: از عمق ریاضیات تا مهندس حرفه‌ای» است.

## محتوای بسته

- `image_processing_next_decade_ch12_fa.tex`: متن کامل فصل به فارسی
- `image_processing_next_decade_ch12_fa.pdf`: نسخه کامپایل‌شده ۶۱ صفحه‌ای
- `figures/`: ۲۵ شکل در دو قالب PDF و PNG
- `generate_figures.py`: بازتولید همه شکل‌ها
- `code/`: کدهای مستقل VAE، VQ-VAE، GAN، WGAN-GP، RealNVP، PixelCNN، FID/KID، درون‌یابی نهفته و Flow Matching
- `code/run_smoke_tests.py`: اجرای یکجای آزمون‌های سریع
- `outputs/smoke_test.txt`: خروجی واقعی آزمون‌ها
- `requirements.txt`: وابستگی‌های پایتون
- `MANIFEST_SHA256.txt`: کنترل صحت فایل‌های بسته

## ساخت PDF

پیش‌نیازها: TeX Live، XeLaTeX و یکی از فونت‌های `Amiri` یا `Noto Naskh Arabic`.

```bash
xelatex image_processing_next_decade_ch12_fa.tex
xelatex image_processing_next_decade_ch12_fa.tex
```

کامپایل باید از پوشه اصلی بسته انجام شود تا مسیرهای `figures/` و `code/` درست resolve شوند.

## اجرای کدها

```bash
python -m venv .venv
source .venv/bin/activate          # Linux/macOS
# .venv\Scripts\activate           # Windows PowerShell
pip install -r requirements.txt
python code/run_smoke_tests.py
```

برای تولید دوباره شکل‌ها:

```bash
python generate_figures.py
```

## نکات علمی بازتولیدپذیری

- FID، KID و precision/recall باید با extractor، resize، تعداد نمونه و آمار مرجع ثبت‌شده گزارش شوند.
- نمونه‌های تصویری منتخب جایگزین نمونه‌گیری تصادفی، failure gallery و nearest-neighbor audit نیستند.
- در VAE باید reconstruction، KL، active units و نمونه‌های prior جداگانه پایش شوند.
- در VQ-VAE، perplexity به‌تنهایی کافی نیست؛ occupancy، کدهای مرده و کیفیت بازسازی نیز لازم‌اند.
- در GAN، loss خام معادل پوشش مدها یا کیفیت نمونه نیست؛ occupancy، precision/recall و چند seed لازم است.
- در جریان‌های نرمال‌ساز، آزمون وارون‌پذیری و تطبیق log-determinant تحلیلی با کنترل عددی ضروری است.

## وضعیت کنترل کیفیت

- PDF: ۶۱ صفحه A4، غیررمزشده
- فونت‌های PDF تعبیه شده‌اند
- همه ۱۰ فایل پایتون با `py_compile` کنترل شده‌اند
- ۹ آزمایش مستقل با موفقیت اجرا شده‌اند
- بسته هیچ فایل فونتی را توزیع نمی‌کند
