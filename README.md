# Типы локальных экономик России: динамическая кластеризация муниципалитетов на атрибутированных сетях

Решение для онлайн-конкурса СберИндекса 2026, направление «Кластеризация».

- **Интерактивный лендинг «Атлас типов»** — https://ivanfrolovskiy.github.io/sberindex-atlas/ и он же файлом: `site/index.html`.
  Это один автономный файл на 3,7 МБ: ни сервера, ни внешних запросов, открывается и по ссылке, и с диска.
  История из семи глав ведёт от карты трат к типам, их динамике и волне маркетплейсов. Дальше интерактивная часть:
  атлас по слоям и месяцам с масштабированием, паспорт любого МО с сопоставимыми территориями, карточки типов, метод.
  Движение привязано к прокрутке, карта переключается кроссфейдом кэшированных битмап. Есть светлая тема.
  Печатная версия — `site/landing.pdf` (35 стр., собирается из лендинга скриптом).
- **Методологический отчёт**: `report/methodology.md` (PDF: `report/methodology.pdf`).
- **Публикация**: страница раздаётся GitHub Pages из корня ветки `main`; корневой `index.html` переадресует на `site/index.html`.
- **Код**: пакет `src/mo_types`, скрипты в `scripts/`, конфигурации в `configs/`, тесты в `tests/`.

## Что сделано

1. Панель 2 016 МО × 24 месяца (СберИндекс, `territory_id`) + Росстат (население, зарплата, занятость по ОКВЭД2) + индекс доступности рынков + дорожные расстояния + полигоны.
2. Сравнение восьми правил ребра (косинус профиля, корреляции и лаговые корреляции приростов, DTW уровня и композиции, дорожная гравитация, SNF-слияние, обученный граф) и пяти способов разрежения; итог — обученный граф из гладких сигналов.
3. KEFRiN (Shalileh & Mirkin, 2022) и его временное расширение **KEFRiN-T** (сглаживание с адаптивным весом истории по AFFECT, фиксированные эффекты узлов, тест инновации), проверенное на синтетике с известной истиной; зоопарк из двенадцати методов (K-means, Ward, GMM, спектральная, Leiden двух разрешений, iLouvain, EVA, CANUS, DMoN, KEFRiN на признаках и на сети).
4. Полный набор ICVI: SW, CH, S_Dbw, AVI, AVU, ANUI, MQ, модульность (сверено с библиотекой Pattern), bootstrap-устойчивость, устойчивость между месяцами.
5. Интерпретация: правило Миркина, пороговые правила IMM-дерева, интервальные описания, матрица переходов и Spatial Markov, тест сходимости log-t, кейсы шоков, «сопоставимые МО».
6. Две линзы: экономическая типология (потребление + контекст) и поведенческая (только СберИндекс, K = 5), кросс-таблица и что даёт каждая половина признаков (`scripts/two_lenses.py`).
7. Внешняя проверка по индексу покупательской мобильности СберИндекса (`scripts/mobility_check.py`) и сетевой взгляд без признаков (`scripts/network_view.py`).
8. Волна маркетплейсов: единственный большой сдвиг структуры трат (доля удвоилась), догоняющий рост села, клубная сходимость внутри типов (`scripts/marketplace_wave.py`).

## Быстрый старт

```bash
make env                       # окружение .venv312 (Python 3.12 через uv: https://docs.astral.sh/uv/) + graph-learning из git
.venv312/bin/python scripts/download_data.py   # исходные данные (архив хакатона, границы МО, Росстат, API)
make all                       # панель → графы → KEFRiN-T → правила ребра → методы → синтетика → устойчивость →
                               # интерпретация → кейсы → итоговый пакет → волна и линзы → лендинг → отчёт
make test
```

Отдельные шаги: `make panel`, `make graphs`, `make kefrin`, `make edge`, `make zoo`, `make synthetic`, `make stability`,
`make interpret`, `make cases`, `make wave`, `make lenses`, `make netview`, `make mobility`, `make cleanup`, `make finalize`,
`make landing`, `make report`. Необязательные: `make first-pass` (разведочный прогон) и `make tgfa` (время-варьирующие
графы, §4 отчёта). Все гиперпараметры — в `configs/base.yaml`
(признаки, графы, K, seed) и `configs/types.yaml` (метки итоговых типологий — экономической, подтипов и поведенческой — и имена типов).

## Структура

```
configs/        base.yaml (пайплайн), types.yaml (итоговая типология)
src/mo_types/   data (панель), features, graphs, learned_graph, edge_rules, clustering (KEFRiN, правило Миркина),
                temporal (KEFRiN-T), metrics (ICVI), interpret (IMM, паттерны, Markov, log-t, сопоставимые МО)
scripts/        download_data, build_panel, first_pass, learned_graph, learn_window_graphs, learn_tgfa_graphs,
                kefrin_t, compare_graphs, analyze_kefrin_t, edge_rules, method_zoo, synthetic_validation,
                stability_k, interpret_typology, case_studies, marketplace_wave, two_lenses, network_view, mobility_check, dynamics_cleanup, finalize,
                prepare_site_data (данные лендинга), build_site (сборка одного файла), build_print (печатная версия),
                build_report (отчёт: md → html → pdf),
                qa_site (проверка лендинга в Chromium и WebKit через Playwright: кадры по шагам, ошибки, статистика кадров)
outputs/        результаты по шагам (csv/json); outputs/final — итоговый пакет типологии
site/           index.html — лендинг, landing.pdf — печатная версия;  site_src/ — исходники лендинга (шаблон, стили, js/, D3, шрифты Inter и Unbounded, данные)
report/         методологический отчёт: methodology.md, methodology.pdf, figures/
tests/          pytest: метрики, временной метод, DMoN, данные лендинга
```

## Данные и лицензии

СберИндекс: потребительские безналичные расходы на уровне МО, индекс доступности рынков, автодорожные связи
(архив хакатона Data → Sense), справочник границ МО — CC BY-SA 4.0. Росстат БД ПМО через каталог
«Если быть точным» — CC BY 4.0. Все пакеты — открытые (Apache/MIT/BSD/GPL); проприетарные платные технологии не используются.

Цитирование данных: «Потребительские безналичные расходы на уровне муниципальных образований. СберИндекс.
https://sberindex.ru/ru/research/data-sense-opisanie-nabora-dannikh-khakatona-sberindeksa-po-munitsipalnim-dannim»;
«Данные о границах и преобразованиях муниципальных образований. СберИндекс.
https://sberindex.ru/ru/research/dataset-borders-and-changes-of-municipalities».

Сторонний код: CANUS (S. Shalileh, https://github.com/Sorooshi/CANUS) не имеет явной лицензии, поэтому в репозиторий
не включён — `make env` скачивает `third_party/canus.py` с проверкой sha256; без него сравнение методов пропускает CANUS.
Обучение графов — пакет `graph-learning` (LTS4, EPFL), ставится из git при `make env`.
В лендинг встроены D3 v7 и d3-sankey (ISC, Mike Bostock) и шрифты Inter и Unbounded (SIL Open Font License 1.1);
исходники — в `site_src/vendor/` и `site_src/fonts/`.

Лицензия кода: MIT.
