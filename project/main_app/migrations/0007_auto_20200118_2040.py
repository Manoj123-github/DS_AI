

from django.db import migrations


class Migration(migrations.Migration):

    dependencies = [
        ('main_app', '0006_remove_consultation_messages'),
    ]

    operations = [
        migrations.RemoveField(
            model_name='doctor',
            name='img',
        ),
        migrations.RemoveField(
            model_name='patient',
            name='img',
        ),
    ]
