

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('main_app', '0002_rating_review'),
    ]

    operations = [
        migrations.AlterField(
            model_name='rating_review',
            name='review',
            field=models.TextField(blank=True),
        ),
    ]
