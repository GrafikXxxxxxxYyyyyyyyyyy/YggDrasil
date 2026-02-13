"""Шаблон новой модальности. Просто скопируй эту папку и переименуй."""

from .modality import CustomModality

# Автоматическая регистрация при импорте
CustomModality.register_blocks()