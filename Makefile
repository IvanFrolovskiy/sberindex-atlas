PY := .venv312/bin/python

.PHONY: all env third_party wave lenses netview mobility cleanup panel graphs kefrin edge zoo synthetic stability interpret cases finalize landing report tgfa first-pass test clean

all: panel graphs kefrin edge zoo synthetic stability interpret cases finalize wave lenses netview mobility cleanup landing report

env: third_party
	uv venv --python 3.12 .venv312
	uv pip install --python $(PY) -r requirements.txt
	uv pip install --python $(PY) "git+https://github.com/LTS4/graph-learning.git"

third_party/canus.py:
	curl -sL -o third_party/canus.py https://raw.githubusercontent.com/Sorooshi/CANUS/main/canus.py
	echo "dbc14daff4f6b2a4b9d377deb1bef5c76a2c390678d47922fa337983e0faa7a3  third_party/canus.py" | shasum -a 256 -c -

third_party: third_party/canus.py

panel:
	$(PY) scripts/build_panel.py

graphs:
	$(PY) scripts/learned_graph.py --static-only
	$(PY) scripts/learn_window_graphs.py

kefrin:
	$(PY) scripts/kefrin_t.py --k 4,5,6,7,8 --tag minsize
	$(PY) scripts/kefrin_t.py --k 4,5,6 --graphs learned --graph-dir outputs/learned_graph/windowlog --tag windowlog --two-pass
	$(PY) scripts/kefrin_t.py --k 4,5,6 --graphs learned --graph-dir outputs/learned_graph/windowlog --tag beh_windowlog --no-context --two-pass
	$(PY) scripts/compare_graphs.py

edge:
	$(PY) scripts/edge_rules.py --B 10

zoo:
	$(PY) scripts/method_zoo.py --graph learned --B 10

synthetic:
	$(PY) scripts/synthetic_validation.py --seeds 5

stability:
	$(PY) scripts/stability_k.py --graph knn --B 20
	$(PY) scripts/stability_k.py --graph learned --B 20 --k 4,5,6,7
	$(PY) scripts/stability_k.py --graph learned --B 20 --k 3,4,5,6,7 --no-context

interpret:
	$(PY) scripts/interpret_typology.py --labels outputs/kefrin_t/labels_affect_fe_K4_learned_windowlog.csv --tag K4_windowlog
	$(PY) scripts/interpret_typology.py --labels outputs/kefrin_t/labels_affect_fe_K6_learned_windowlog.csv --tag K6_windowlog
	$(PY) scripts/interpret_typology.py --labels outputs/kefrin_t/labels_affect_fe_K5_learned_beh_windowlog.csv --tag K5_beh_windowlog
	$(PY) scripts/analyze_kefrin_t.py --labels outputs/kefrin_t/labels_affect_fe_K4_learned_windowlog.csv --alphas outputs/kefrin_t/alphas_learned_windowlog.csv

cases:
	$(PY) scripts/case_studies.py --labels outputs/kefrin_t/labels_affect_fe_K4_learned_windowlog.csv

wave:
	$(PY) scripts/marketplace_wave.py

lenses:
	$(PY) scripts/two_lenses.py

netview:
	$(PY) scripts/network_view.py

mobility:
	$(PY) scripts/mobility_check.py

# сравнение первого и второго прохода; снимок первого лежит в outputs/dynamics_cleanup/baseline
cleanup:
	$(PY) scripts/dynamics_cleanup.py

finalize:
	$(PY) scripts/finalize.py

landing:
	$(PY) scripts/prepare_site_data.py
	$(PY) scripts/build_site.py
	$(PY) scripts/build_print.py

report:
	$(PY) scripts/build_report.py

# необязательные шаги: разведочный прогон и время-варьирующие графы (§4 отчёта)
first-pass:
	$(PY) scripts/first_pass.py

tgfa:
	$(PY) scripts/learn_tgfa_graphs.py

test:
	$(PY) -m pytest -q tests

# снимок первого прохода (outputs/dynamics_cleanup/baseline) не удаляется: это вход сравнения в make cleanup
clean:
	rm -rf outputs/first_pass outputs/kefrin_t outputs/learned_graph outputs/edge_rules outputs/method_zoo outputs/synthetic outputs/stability outputs/interpretation outputs/final
