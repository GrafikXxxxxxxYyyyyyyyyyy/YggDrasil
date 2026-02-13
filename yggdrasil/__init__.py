# yggdrasil/__init__.py
"""YggDrasil — Lego-конструктор диффузии."""

# Авто-регистрация ключевых блоков при импорте yggdrasil
import yggdrasil.core.block.registry
from yggdrasil.core.block.registry import auto_discover

auto_discover()  # ← если у тебя есть такая функция в registry.py