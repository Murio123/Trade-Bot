"""Offline-инструменты (вне production runtime).

Пакет намеренно НЕ импортируется боевым кодом (scheduler/pipeline/handlers/
main). Живёт вне guard-списка test_production_code_does_not_import_contracts,
поэтому его модулям разрешено импортировать contracts для shadow-проверок.
"""
