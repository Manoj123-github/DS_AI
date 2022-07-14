

from django.db import migrations


class Migration(migrations.Migration):

    dependencies = [
        ('main_app', '0005_doctor_rating'),
    ]

    operations = [
        migrations.RemoveField(
            model_name='consultation',
            name='messages',
        ),
    ]
