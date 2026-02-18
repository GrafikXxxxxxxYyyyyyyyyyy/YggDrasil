# Как добавить новую модальность за 30 минут

1. `cp -r plugins/custom plugins/my_awesome_modality`
2. Переименуй все `Custom` → `MyAwesome`
3. Отредактируй 3–4 файла:
   - `modality.py` (to_tensor, visualize, get_default_pipeline)
   - `backbone.py` (твоя архитектура)
   - `codec.py` (если нужен латент)
4. Добавь в `plugins/__init__.py` строку импорта (если не используешь auto-discovery)
5. Запусти:
   ```python
   from YggDrasil.plugins.my_awesome_modality.modality import MyAwesomeModality
   model = MyAwesomeModality.create_model()