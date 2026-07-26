# التسليم النهائي — MissionGuard AI مع بيانات OPSSAT الحقيقية

## الإضافات الموجودة في هذه النسخة

- ربط المشروع ببيانات OPSSAT-AD الحقيقية فقط.
- حفظ الموديلات والـPreprocessing داخل ملفات جاهزة، ولذلك الموقع لا يعيد التدريب عند كل تشغيل.
- إضافة ملفات منفصلة للـIsolation Forest والـRandom Forest وملف أسماء الـFeatures.
- تجهيز Train وValidation وTest داخل `data/opssat/processed/` مع الحفاظ على الـOfficial Test Split.
- التحقق من ملفات Upload قبل التنبؤ: الأعمدة، القيم الرقمية، التوقيت، التكرار، القنوات، والـLabels.
- إضافة Ground-Truth Evaluation عند وجود Labels.
- إضافة Event-Based Evaluation وEvent Detection Ledger.
- إضافة Telemetry Drift Monitor لمقارنة البيانات الجديدة ببيانات التدريب الطبيعية.
- إضافة `data/DATASET_CARD.md` لتوثيق المصدر والترخيص والمعالجة والقيود.
- تجهيز خمسة ملفات Upload حقيقية، منها Normal → Anomaly → Recovery وأمثلة من Magnetometer وPhotodiode.
- الحفاظ على Light Mode وDark Mode عاليي التباين.

## تشغيل المشروع

```powershell
cd MissionGuard_AI
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

أو:

```powershell
RUN_WINDOWS.bat
```

## إعادة تدريب الموديلات عند الحاجة فقط

```powershell
python train_models.py
```

الأمر السابق يعيد إنشاء:

- الموديل المدمج.
- ملفات الموديلات المنفصلة.
- Train / Validation / Test processed files.
- نتائج الـOfficial Test.
- Metadata وFeature schema.

## تجربة ملفات الرفع

اختاري من القائمة الجانبية:

```text
Upload Real OPSSAT CSV
```

ثم ارفعي ملفًا من:

```text
data/opssat/upload_samples/
```

أفضل ملف للعرض:

```text
opssat_real_mixed.csv
```

ويحتوي على تسلسل حقيقي:

```text
Normal → Anomaly → Normal/Recovery
```

## ملاحظة علمية

MissionGuard AI يكتشف Statistical Telemetry Anomalies ولا يثبت سببًا هندسيًا مؤكدًا للعطل. الـAnomaly Score ليس Failure Probability معتمدة، وأي قرار تشغيلي يحتاج مراجعة بشرية.


## تشغيل PostgreSQL وpgAdmin على سيرفر

النسخة الحالية تحتوي على `docker-compose.yml` يشغّل التطبيق وقاعدة PostgreSQL
وواجهة pgAdmin معًا. انسخي `.env.server.example` إلى `.env`، غيّري كلمات المرور،
ثم شغّلي:

```bash
docker compose up -d --build
```

التطبيق ينشئ Schema باسم `missionguard` ويجهز الجداول وبيانات OPS-SAT الأساسية
تلقائيًا بدون تكرارها عند كل Restart. الشرح الكامل موجود في:

```text
DEPLOYMENT_PGADMIN_AR.md
```
