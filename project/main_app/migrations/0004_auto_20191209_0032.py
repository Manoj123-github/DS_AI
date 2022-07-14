

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('main_app', '0003_auto_20191208_2155'),
    ]

    operations = [
        migrations.AlterField(
            model_name='rating_review',
            name='rating',
            field=models.IntegerField(default=0),
        ),
    ]
