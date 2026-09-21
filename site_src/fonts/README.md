# Шрифты

Подмножества кириллицы и латиницы в формате woff2; собираются в лендинг как data-URI (`scripts/build_site.py`),
поэтому страница не обращается к внешним серверам.

- `inter-*.woff2` — Inter, The Inter Project Authors. SIL Open Font License 1.1, текст в `LICENSE-Inter.txt`,
  https://github.com/rsms/inter
- `unbounded-*.woff2` — Unbounded, The Unbounded Project Authors. SIL Open Font License 1.1, текст
  в `LICENSE-Unbounded.txt`, https://github.com/googlefonts/unbounded

`subsets.json` — перечень файлов с диапазонами символов, по нему собираются правила @font-face.
