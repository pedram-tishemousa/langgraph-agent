import sqlite3

# ۱. ساخت یا اتصال به فایل دیتابیس
conn = sqlite3.connect('company.db')
cursor = conn.cursor()

# ۲. ساخت یک جدول برای سرورها
cursor.execute('''
CREATE TABLE IF NOT EXISTS servers (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    ip_address TEXT,
    status TEXT,
    cpu_load TEXT
)
''')

# ۳. پاک کردن دیتای قبلی (برای جلوگیری از تکرار در صورت اجرای مجدد)
cursor.execute('DELETE FROM servers')

# ۴. تزریق داده‌های نمونه
servers_data = [
    ('web', '192.168.1.10', 'Online', '45%'),
    ('database', '192.168.1.20', 'Offline', '0%'),
    ('cache', '192.168.1.30', 'Online', '88%')
]

cursor.executemany('''
INSERT INTO servers (name, ip_address, status, cpu_load) 
VALUES (?, ?, ?, ?)
''', servers_data)

# ۵. ذخیره و بستن
conn.commit()
conn.close()

print("✅ دیتابیس با موفقیت ساخته شد و داده‌ها ثبت شدند.")