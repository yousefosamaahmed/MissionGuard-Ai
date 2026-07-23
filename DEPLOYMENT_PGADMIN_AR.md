# تشغيل MissionGuard AI على سيرفر مع PostgreSQL وpgAdmin

هذه النسخة مجهزة بثلاث خدمات داخل Docker Compose:

- `app`: تطبيق Streamlit.
- `postgres`: قاعدة بيانات PostgreSQL دائمة.
- `pgadmin`: واجهة pgAdmin لإدارة الجداول والبيانات.

## 1) إعداد كلمات المرور

نسخة Windows المحلية المرفقة تحتوي ملف `.env` جاهزًا للتشغيل مباشرة. القيم الافتراضية محلية فقط ومذكورة في `RUN_WITH_PGADMIN_AR.txt`. لا ترفعي `.env` إلى GitHub.

عند النشر على سيرفر، استخدمي بدلًا منه:

```bash
cp .env.server.example .env
```

ثم غيّري كل قيمة تحتوي على `CHANGE_ME` إلى كلمات مرور قوية ومختلفة.

## 2) التشغيل على جهازك

```bash
docker compose up -d --build --force-recreate
```

أو على Windows:

```text
START_DOCKER_WINDOWS.bat
```

بعد اكتمال التشغيل:

- التطبيق: `http://localhost:8501`
- pgAdmin: `http://localhost:5050`
- PostgreSQL من Windows: `127.0.0.1:55432` (داخل Docker يظل `5432`)

سجلي الدخول إلى pgAdmin بالقيمتين:

- `PGADMIN_DEFAULT_EMAIL`
- `PGADMIN_DEFAULT_PASSWORD`

ستجدي سيرفرًا محفوظًا باسم **MissionGuard PostgreSQL**. عند أول فتح له، اكتبي كلمة مرور `POSTGRES_PASSWORD`.

مسار الجداول داخل pgAdmin:

```text
Servers
└── MissionGuard PostgreSQL
    └── Databases
        └── missionguard_ai
            └── Schemas
                └── missionguard
                    └── Tables
```


## حل تعارض منفذ PostgreSQL على Windows

هذه النسخة تستخدم المنفذ `55432` على Windows بدلًا من `5432` لتجنب التعارض مع PostgreSQL محلي أو منفذ محجوز. لا تغيّري `POSTGRES_PORT=5432` لأن التطبيق وpgAdmin يتصلان بقاعدة البيانات داخل شبكة Docker على `postgres:5432`. المتغير الذي يتحكم في منفذ Windows فقط هو:

```env
POSTGRES_EXPOSE_PORT=55432
```

بعد أي تعديل شغّلي:

```powershell
docker compose down --remove-orphans
docker compose up -d --build
docker compose ps
```

## 3) الرفع على Ubuntu VPS

ارفعي مجلد المشروع إلى السيرفر، ثم من داخل المجلد:

```bash
cp .env.server.example .env
nano .env
chmod +x START_SERVER_LINUX.sh
./START_SERVER_LINUX.sh
```

التطبيق يكون متاحًا على:

```text
http://SERVER_IP:8501
```

افتحي المنفذ `8501/tcp` فقط في Firewall أو Security Group. قاعدة PostgreSQL وpgAdmin مربوطتان بـ`127.0.0.1` افتراضيًا، لذلك لا تكونان مكشوفتين مباشرة على الإنترنت.

## 4) فتح pgAdmin على السيرفر بأمان

من جهازك شغلي SSH Tunnel:

```bash
ssh -L 5050:127.0.0.1:5050 USER@SERVER_IP
```

ثم افتحي:

```text
http://127.0.0.1:5050
```

لا يُنصح بتغيير `PGADMIN_BIND_ADDRESS` إلى `0.0.0.0` إلا عند وجود HTTPS وReverse Proxy وحماية مناسبة.

## 5) أوامر المتابعة

حالة الخدمات:

```bash
docker compose ps
```

عرض السجلات:

```bash
docker compose logs -f app
```

إعادة التشغيل:

```bash
docker compose restart
```

تحديث النسخة بعد رفع كود جديد:

```bash
docker compose up -d --build
```

إيقاف الخدمات مع الاحتفاظ بالداتا:

```bash
docker compose down
```

حذف الخدمات والداتا نهائيًا:

```bash
docker compose down -v
```

الأمر الأخير يحذف قاعدة البيانات، لذلك لا تستخدميه إلا عند التأكد.

## 6) النسخ الاحتياطي والاسترجاع

نسخة احتياطية:

```bash
docker compose exec -T postgres pg_dump \
  -U missionguard \
  -d missionguard_ai \
  -Fc > missionguard_backup.dump
```

استرجاع نسخة:

```bash
docker compose exec -T postgres pg_restore \
  -U missionguard \
  -d missionguard_ai \
  --clean --if-exists < missionguard_backup.dump
```

إذا غيّرتِ `POSTGRES_USER` أو `POSTGRES_DB` في `.env`، استخدمي القيم الجديدة في أوامر النسخ الاحتياطي.

## 7) قاعدة بيانات Cloud خارج Docker

الكود يدعم `DATABASE_URL` مباشرة، مثل:

```env
DATABASE_URL=postgresql+psycopg://USER:PASSWORD@HOST:5432/DBNAME?sslmode=require
POSTGRES_SCHEMA=missionguard
```

متغير `DATABASE_URL` له الأولوية على متغيرات `POSTGRES_USER` و`POSTGRES_HOST` وغيرها. لا تضعي كلمة المرور داخل GitHub أو داخل الكود؛ ضعيها في Environment Variables الخاصة بالسيرفر.
