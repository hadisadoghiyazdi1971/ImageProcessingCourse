# ممیزی منابع فصل دوم

این فایل مشخص می‌کند کدام بخش‌های فصل بر منابع بنیادی، استاندارد رسمی یا مقاله پژوهشی اولیه تکیه دارند. منابع وبلاگی و صفحات خلاصه‌ساز مبنای روابط فصل نبوده‌اند.

| محور | منبع اولیه/رسمی | نوع | استفاده در فصل |
|---|---|---|---|
| نظریه نمونه‌برداری مدرن | M. Unser, “Sampling—50 Years after Shannon,” Proceedings of the IEEE, 2000 | مقاله مروری مرجع در IEEE | محدودیت باند‌محدودبودن ایده‌آل، نمونه‌برداری تعمیم‌یافته و دستگاه اخذ غیرایده‌آل |
| نمونه‌برداری تعمیم‌یافته | A. Papoulis, “Generalized Sampling Expansion,” IEEE TCAS, 1977 | مقاله بنیادی | اندازه‌گیری‌های چندکاناله و تعمیم نمونه نقطه‌ای |
| کوانتیزه‌سازی بهینه | J. Max, 1960; S. Lloyd, 1982 | مقالات بنیادی IEEE | شروط مرکز جرم و مرزهای تصمیم Lloyd–Max |
| مشخصه‌سنجی حسگر | EMVA Standard 1288, Release 4.0 | استاندارد رسمی | نویز، SNR، اشباع، حساسیت و دامنه دینامیکی دوربین |
| ناظر رنگ‌سنجی | ISO/CIE 11664-1:2019 | استاندارد رسمی CIE/ISO | توابع تطبیق رنگ ناظر استاندارد CIE 1931 |
| sRGB | IEC 61966-2-1 | استاندارد رسمی IEC | تابع انتقال قطعه‌ای sRGB و خطی‌سازی |
| UHD/WCG | ITU-R BT.2020 | توصیه‌نامه رسمی ITU | اولیه‌ها و پهنه رنگ UHD |
| HDR | ITU-R BT.2100-3, February 2025 | آخرین نسخه رسمی ITU در زمان نگارش | PQ، HLG، پارامترهای HDR و رنگ‌سنجی BT.2020 |
| پاسخ دوربین و HDR چندنوردهی | Debevec & Malik, SIGGRAPH 1997 | مقاله بنیادی | بازیابی تابع پاسخ و نقشه تابش |
| فضای توابع پاسخ دوربین | Grossberg & Nayar, CVPR 2003 | مقاله اولیه | رد فرض یک گامای ثابت برای همه دوربین‌ها |
| آرایه بایر | B. Bayer, U.S. Patent 3,971,065, 1976 | سند اولیه اختراع | مدل نمونه‌برداری RGGB |
| دموزاییک کلاسیک | Malvar, He, Cutler, ICASSP 2004 | مقاله اولیه | درون‌یابی خطی با اصلاح گرادیانی |
| نویز واقعی RAW | Brooks et al., CVPR 2019 | مقاله اولیه CVF | unprocessing و مدل پواسون ـ گاوسی در فضای خام |
| ضدعلیاسینگ در CNN | R. Zhang, ICML 2019 | مقاله اولیه PMLR | مشکل stride/pooling و بهبود shift invariance |
| ضدعلیاسینگ در مولدها | Karras et al., NeurIPS 2021 | مقاله اولیه | طراحی پیوسته و alias-free در GAN |
| ضدعلیاسینگ در دیفیوژن | Zhou et al., CVPR 2025 | مقاله اولیه CVF | shift equivariance کسری در فضای نهفته مدل دیفیوژن |
| داده فراطیفی | ARAD_1K / NTIRE 2022 | مقاله و داده اولیه CVF | ۱۰۰۰ تصویر طبیعی، ۲۰۴ باند اصلی و نسخه ۳۱ باندی چالش |

## نشانی‌های رسمی برای کنترل نسخه

- IEEE sampling paper: https://ieeexplore.ieee.org/document/843002/
- ITU-R BT.2100 current record: https://www.itu.int/rec/r-rec-bt.2100
- ITU-R BT.2100-3 PDF: https://www.itu.int/dms_pubrec/itu-r/rec/bt/R-REC-BT.2100-3-202502-I!!PDF-E.pdf
- CIE standard observer: https://www.cie.co.at/publications/colorimetry-part-1-cie-standard-colorimetric-observers-0
- EMVA 1288: https://www.emva.org/standards-technology/emva-1288/
- Unprocessing Images: https://openaccess.thecvf.com/content_CVPR_2019/html/Brooks_Unprocessing_Images_for_Learned_Raw_Denoising_CVPR_2019_paper.html
- ARAD_1K challenge paper: https://openaccess.thecvf.com/content/CVPR2022W/NTIRE/html/Arad_NTIRE_2022_Spectral_Recovery_Challenge_and_Data_Set_CVPRW_2022_paper.html
- Alias-Free LDM: https://openaccess.thecvf.com/content/CVPR2025/html/Zhou_Alias-Free_Latent_Diffusion_Models_Improving_Fractional_Shift_Equivariance_of_Diffusion_CVPR_2025_paper.html
- Shift-invariant CNN: https://proceedings.mlr.press/v97/zhang19a.html

## مرز ادعاها

- رابطه واریانس کوانتیزه‌سازی `Delta^2/12` فقط در تقریب پُروضوح، بدون اشباع و با چگالی تقریباً ثابت در هر سلول استفاده شده است.
- رابطه `6.02b + 1.76 dB` برای سینوس تمام‌مقیاس و ADC ایده‌آل است، نه دامنه دینامیکی قطعی یک دوربین.
- بازسازی طیف از RGB به‌عنوان مسئله بدوضع معرفی شده است؛ نتایج شبکه آموخته‌شده به‌عنوان بازیابی یکتای فیزیکی تفسیر نشده‌اند.
- PQ و HLG با نقش‌های استانداردی متفاوت توضیح داده شده‌اند؛ نمودار مشترک فقط تخصیص کد را مقایسه می‌کند.
