# ممیزی منابع فصل سوم

**تاریخ بررسی وب:** 2026-07-21  
**اصل انتخاب:** کتاب مرجع معتبر، مقاله اصلی به‌جای خلاصه ثانویه، صفحه رسمی ناشر/کنفرانس/استاندارد، و تفکیک روشن میان قضیه ریاضی، مدل فیزیکی و نتیجه تجربی.

## منابع بنیادی و کتاب‌ها

| منبع | نوع و اعتبار | نقش دقیق در فصل | شناسه یا صفحه رسمی |
|---|---|---|---|
| Bracewell, *The Fourier Transform and Its Applications*, 3rd ed. | کتاب مرجع تحلیل فوریه | قرارداد تبدیل، خواص، تفسیر مکانی-فرکانسی | ISBN 9780073039381 |
| Oppenheim, Schafer, Buck, *Discrete-Time Signal Processing* | کتاب مرجع DSP | DFT، تناوب، پنجره، نشت، FFT و بانک فیلتر | ISBN 9780137549207 |
| Mallat, *A Wavelet Tour of Signal Processing*, 3rd ed. | کتاب مرجع موجک و تنکی | MRA، بانک فیلتر، چارچوب و scattering | https://www.sciencedirect.com/book/9780123743701/a-wavelet-tour-of-signal-processing |
| Daubechies, *Ten Lectures on Wavelets* | تک‌نگاشت SIAM | پایه‌های متعامد، گشتاور محوشونده و مصالحه تکیه‌گاه-همواری | https://doi.org/10.1137/1.9781611970104 |
| Gonzalez and Woods, *Digital Image Processing*, 4th ed. | کتاب استاندارد پردازش تصویر | فیلترگذاری مکانی و فرکانسی، PSF/OTF/MTF | ISBN 9780133356724 |
| Szeliski, *Computer Vision: Algorithms and Applications*, 2nd ed. | کتاب روز و آزاد نویسنده | اتصال هرم‌ها و فیلترها به بینایی محاسباتی | https://szeliski.org/Book/ |
| Hansen, Nagy, O'Leary, *Deblurring Images* | تک‌نگاشت SIAM | BTTB/BCCB، شرط مرزی، طیف و مسئله معکوس رفع تاری | https://doi.org/10.1137/1.9780898718874 |
| Lindeberg, *Scale-Space Theory in Computer Vision* | تک‌نگاشت تخصصی | اصول فضای مقیاس و انتخاب خودکار مقیاس | https://doi.org/10.1007/978-1-4757-6465-9 |

## مقالات اصلی کلاسیک

| مقاله | ادعای استفاده‌شده | صفحه رسمی/DOI |
|---|---|---|
| Cooley and Tukey (1965) | تجزیه radix-2 و کاهش مرتبه محاسبات DFT | https://doi.org/10.1090/S0025-5718-1965-0178586-1 |
| Gabor (1946) | اتم‌های موضعی زمان-فرکانس و زمینه نظری پنجره گابور | https://doi.org/10.1049/ji-3-2.1946.0074 |
| Marr and Hildreth (1980) | مشتق دوم گاوسی و آشکارسازی لبه چندمقیاسی | https://doi.org/10.1098/rspb.1980.0020 |
| Koenderink (1984) | ساختار تصویر در مقیاس و پیوند گاوسی با انتشار گرما | https://doi.org/10.1007/BF00336961 |
| Burt and Adelson (1983) | هرم لاپلاسی به‌عنوان کد فشرده چندمقیاسی | https://doi.org/10.1109/TCOM.1983.1095851 |
| Mallat (1989) | نظریه MRA و معادل‌سازی موجک با بانک فیلتر | https://doi.org/10.1109/34.192463 |
| Daubechies (1988) | وجود پایه‌های موجک متعامد با تکیه‌گاه فشرده | https://doi.org/10.1002/cpa.3160410705 |
| Freeman and Adelson (1991) | طراحی و ترکیب فیلترهای جهت‌پذیر | https://doi.org/10.1109/34.93808 |
| Simoncelli et al. (1992) | تبدیل‌های چندمقیاسی shiftable و افزونگی کنترل‌شده | https://doi.org/10.1109/18.119725 |
| Simoncelli and Freeman (1995) | معماری هرم steerable | https://www.cns.nyu.edu/pub/eero/simoncelli95b.pdf |
| Candès and Donoho (2004) | curvelet و تقریب بهینه‌تر لبه‌های خمیده قطعه‌ای هموار | https://doi.org/10.1002/cpa.10116 |
| Mallat (2012) | scattering ناوردا و پایدار نسبت به تغییرشکل | https://doi.org/10.1002/cpa.21413 |
| Portilla and Simoncelli (2000) | مدل بافت بر پایه آمار مشترک ضرایب موجک مختلط | https://doi.org/10.1023/A:1026553619983 |

## استاندارد و ابزار بازتولید

| مرجع | وضعیت/کاربرد | صفحه رسمی |
|---|---|---|
| ISO 12233:2024 | نسخه منتشرشده پایه (Edition 5، سپتامبر 2024) برای تفکیک و پاسخ فرکانس مکانی دوربین؛ یک Amendment 1 در 2026 در دست توسعه است. در فصل فقط مبانی ESF/LSF/MTF و آزمایش آموزشی ساده‌شده استفاده شده است | https://www.iso.org/standard/88626.html |
| van der Walt et al. (2014), scikit-image | منبع علمی کتابخانه و تصاویر نمونه آزمایش | https://doi.org/10.7717/peerj.453 |

## مقالات پیونددهنده به یادگیری عمیق و دهه آینده

| مقاله | نقش در فصل | صفحه رسمی |
|---|---|---|
| Zhang (ICML 2019) | نشان‌دادن نقش پایین‌گذر پیش از downsampling در بهبود shift consistency شبکه | https://proceedings.mlr.press/v97/zhang19a.html |
| Rahaman et al. (ICML 2019) | صورت‌بندی و شواهد سوگیری طیفی شبکه‌های عصبی | https://proceedings.mlr.press/v97/rahaman19a.html |
| Tancik et al. (NeurIPS 2020) | Fourier Features و تغییر پهنای باند کرنل مؤثر MLP | https://proceedings.neurips.cc/paper/2020/hash/55053683268957697aa39fba6f231c68-Abstract.html |
| Sitzmann et al. (NeurIPS 2020) | SIREN و نمایش ضمنی با فعال‌ساز تناوبی | https://proceedings.neurips.cc/paper/2020/hash/53c04118df112c13a8c34b38343b9c10-Abstract.html |
| Li et al. (ICLR 2021) | Fourier Neural Operator برای یادگیری نگاشت میان فضاهای تابعی | https://openreview.net/forum?id=c8P9NQVtmnO |
| Karras et al. (NeurIPS 2021) | معماری مولد alias-free و تعریف پیوسته لایه‌ها برای هم‌وردایی بهتر | https://proceedings.neurips.cc/paper/2021/hash/076ccd93ad68be51f23707988e934906-Abstract.html |
| Lindeberg (2020/2022) | شبکه مشتق گاوسی scale-covariant و پیوند اصول فضای مقیاس با معماری شبکه | https://arxiv.org/abs/2011.14759 |
| Unser and Chenouard (SIIMS 2013) | چارچوب پارامتری یکپارچه برای موجک‌های steerable دوبعدی | https://doi.org/10.1137/120866544 |

## کنترل ادعاها

1. عبارت «FFT یک تبدیل جدید نیست» بر تعریف الگوریتم Cooley-Tukey و یکسان‌بودن خروجی آن با DFT تکیه دارد.
2. ادعای یکتایی گاوسی در فضای مقیاس فقط با ذکر مجموعه اصول و دامنه اعتبار مطرح شده و به‌صورت شعار عمومی تعمیم داده نشده است.
3. اثر شرط مرزی با آزمایش واقعی گزارش شده و کانولوشن دوری با کانولوشن خطی خلط نشده است.
4. اعداد جدول آزمایش از `experiment_results.csv` به LaTeX تزریق می‌شوند و نمونه ساختگیِ دستی نیستند.
5. آزمایش لبه مایل برای آموزش زنجیره ESF→LSF→MTF است؛ اندازه‌گیری رسمی استاندارد به تجهیزات، هندسه، oversampling، پردازش و گزارش‌گری کامل استاندارد نیاز دارد.
6. نتایج شبکه کوچک سوگیری طیفی، بازتولید آموزشی یک پدیده‌اند و جایگزین نتایج آماری مقاله اصلی نیستند.
