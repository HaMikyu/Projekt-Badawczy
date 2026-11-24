# Instrukcja dla agenta: douczenie System One na GPU

Ten plik jest dla agenta (Claude Code / Cursor) uruchomionego na komputerze z kartą
NVIDIA. Przeczytaj go w całości, zanim cokolwiek zaczniesz trenować.

## Cel

Zmniejszyć liczbę **fałszywych alarmów na ludzkich esejach akademickich** (ludzki
esej oznaczony jako AI) bez wyraźnej utraty wykrywalności tekstów AI.

Stan wyjściowy (System One v3, aplikacja ocenia tekst fragmentami i uśrednia wynik):

| zbiór | ludzkie → AI | niepewne | AI wykryte |
|---|---|---|---|
| dev (`data/dev/dev.csv`, próg 0,5) | 3,4% | – | 84,2% (ChatGPT 93%, DeepSeek 84%, Llama/Mistral ~73%) |
| Ghostbuster OOD (`data/prepared/ood.csv`) | 10% (TNR 0,900) | – | 99,3% |
| **finalny test `essays_2021/`** | **7,0%** | 12,7% | – (same ludzkie) |

Główna przyczyna błędów: model był trenowany na **pierwszych 384 tokenach** eseju,
czyli głównie na wstępach. Na środkowych fragmentach jest słabo skalibrowany
(daje ~0,00 albo ~0,98), a wstępy i zakończenia ludzkich esejów akademickich
(IvyPanda / StudyCorgi) często wyglądają dla niego jak AI.

## Twarde zasady (nie łam ich)

1. **`essays_2021/` to wyłącznie finalny test.** Nie trenuj na nim, nie dobieraj
   na nim progów, rozmiaru fragmentów, epok ani niczego innego. Uruchom
   `train/final_test_essays_2021.py` **raz**, na samym końcu, dla kandydata, który
   już przeszedł kryteria na dev i OOD. Raportuj wynik uczciwie, nawet jeśli
   jest gorszy.
2. **Dev nie może przeciec do treningu.** `data/dev/dev.csv` (z
   `train/build_dev_set.py`, seed 7) służy do strojenia i wyboru modelu. Nowe
   dane treningowe z IvyPanda muszą wykluczać teksty z dev **i ich tematy**
   (tytuły). Wszystko, co pokrywa się z `essays_2021` (tytuł albo >20 wspólnych
   8-gramów), też odrzucaj; funkcję `shingles` znajdziesz w `build_dev_set.py`.
3. **Nie nadpisuj `models/system-one-final/`**, dopóki kandydat nie spełni
   kryteriów akceptacji. Eksportuj najpierw do `models/system-one-v4/`.
4. Pracuj na **osobnej gałęzi** (np. `system-one-v4`), nie na `master`.
   `.pt` / `.safetensors` idą przez Git LFS (`.gitattributes` już to ustawia).
   Nie commituj `data/` ani `train/runs/` (są w `.gitignore`).

## 1. Środowisko

```powershell
python -m venv .venv
.\.venv\Scripts\pip install torch --index-url https://download.pytorch.org/whl/cu124
.\.venv\Scripts\pip install -r requirements.txt
.\.venv\Scripts\python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
git lfs install; git lfs pull
```

Dobierz wersję CUDA w URL do sterownika (`nvidia-smi`). Jeśli `cuda.is_available()`
zwraca `False`, nie trenuj na CPU, tylko napraw instalację.

## 2. Dane

Folder `data/` nie jest w repo. Ścieżki, których oczekuje `train/build_data.py`:

| plik | źródło |
|---|---|
| `data/Training_Essay_Data.csv` | Kaggle „LLM - Detect AI Generated Text” (Training_Essay_Data) |
| `data/daigt_v3/train.parquet` | DAIGT v3 (Kaggle) |
| `data/argugpt/all.parquet` | ArguGPT |
| `data/ghostbuster/data/{train,test}-00000-of-00001.parquet` | Ghostbuster (HF, układ plików parquet) |
| `data/pile/data/train-0000{0,5}-of-00007-*.parquet` | prawdopodobnie `artem9k/ai-text-detection-pile` (HF) |

Jeśli czegoś brakuje, zapytaj użytkownika: oryginalne pliki ma autor treningu v3.
Nie zgaduj innych zbiorów w zastępstwie, bo zmienisz rozkład danych.

Potem:

```powershell
.\.venv\Scripts\python train\build_data.py      # data/prepared/{train,val,test,ood}.csv
.\.venv\Scripts\python train\build_dev_set.py   # data/dev/dev.csv (jeśli go nie ma)
.\.venv\Scripts\python train\eval_dev.py        # punkt odniesienia v3: zapisz wynik

```

### Nowe dane do v4 (napisz `train/build_data_v4.py`)

1. **Ludzkie eseje akademickie z IvyPanda** (`qwedsacf/ivypanda-essays`, ok. 128 tys.,
   kolumna `TEXT`; plik `ivypanda.parquet` ma ok. 650 MB, więc pobierz go raz do `data/ivypanda/`).
   Weź np. 10–15 tys. esejów, usuń tytuł z pierwszej linii i sekcję bibliografii
   (`strip_human` w `build_dev_set.py`). Wyklucz teksty i tytuły z dev oraz
   wszystko, co pokrywa się z `essays_2021`.
2. **Eseje AI na te same tematy** (`artfultom/ivypanda-llm-generated-essays`, pliki
   CSV `chatgpt_/deepseek_/llama_/mistral_essays_*.csv`, kolumny `title`, `text`,
   `model`). Wyklucz tematy z dev. Utrzymaj balans klas w obrębie tej domeny,
   inaczej model nauczy się „IvyPanda = człowiek”.
3. **Trening na fragmentach, nie tylko na początku tekstu.** Dla każdego eseju
   (także ze starych źródeł) losuj 1–3 okna wyrównane do zdań o długości
   200–380 tokenów: jedno z początku, a pozostałe ze środka i końca. Etykieta
   okna = etykieta eseju. Użyj `split_chunks` z `app/helpers.py`, żeby
   trening widział to samo co aplikacja.
4. Doklej to do puli z `build_data.py`, zrób length-matching i podział
   train/val/test (seed 42). Zbiory `ood.csv` (Ghostbuster test) i `dev.csv`
   zostają nietknięte.

## 3. Trening

Skopiuj `train/train_system_one.py` do `train/train_system_one_v4.py`, ustaw
`DATA_DIR = data/prepared_v4` i `RUN_DIR = train/runs/system-one-v4`.
Startuj od wag v3 (szybciej i stabilniej):

```python
model = SystemOneOptionScorer(MODEL_NAME, LR, ...)
bundle = torch.load("models/system-one-final/system_one.pt", map_location="cpu", weights_only=False)
model.load_state_dict(bundle["state_dict"])
```

Proponowane ustawienia: lr `1e-5`, 2–3 epoki, batch 16–32 (ile zmieści VRAM
przy `max_length=384`, precision `16-mixed`), `CLASS_WEIGHTS` bez zmian. Jeśli
masz czas, porównaj 2–3 warianty: (a) bez okien, tylko z danymi IvyPanda,
(b) z oknami, (c) z oknami i lr `2e-5`. Każdy wariant zapisuj do osobnego `RUN_DIR`.

Eksport kandydata (zmień w skrypcie `RUN_DIR` i `OUT = models/system-one-v4`):

```powershell
.\.venv\Scripts\python train\export_system_one.py
```

## 4. Ocena i kryteria akceptacji

Dla każdego kandydata:

```powershell
.\.venv\Scripts\python train\eval_dev.py models\system-one-v4
```

oraz OOD Ghostbuster (`data/prepared/ood.csv`, metryki z `train_system_one.py`).

Kandydat przechodzi, gdy **jednocześnie**:

- dev, próg 0,5: ludzkie → AI **≤ 2%** i AI wykryte **≥ 84%**
  (nie gorzej niż v3; porównuj z wynikiem `eval_dev.py` dla v3 z punktu 2),
- Ghostbuster OOD: accuracy **≥ 0,94**,
- żaden generator w `by_source_at_0.5` nie spada o więcej niż 5 p.p. względem v3.

Jeśli zmieniasz progi aplikacji (`AI_THRESHOLD`, `HUMAN_THRESHOLD`, `CHUNK_TOKENS`
w `app/helpers.py`), dobieraj je **tylko na dev**.

## 5. Finalny test (raz)

Dopiero dla kandydata, który przeszedł punkt 4:

```powershell
$env:DETEKTOR_SYSTEM_ONE_DIR = "models\system-one-v4"
.\.venv\Scripts\python train\final_test_essays_2021.py
```

Porównaj z v3: 80,3% człowiek / 12,7% niepewne / 7,0% AI. Jeśli v4 jest lepszy,
skopiuj go do `models/system-one-final/`, zaktualizuj `train/WYNIKI.md`, `README.md`
i rozdział 6 dokumentacji (`Dokumentacja/sections/60-wyniki.tex`, potem
`Dokumentacja/compile.sh`). Jeśli jest gorszy, zostaw v3 i opisz, co nie zadziałało.
Nie poprawiaj modelu po obejrzeniu wyniku finalnego testu i nie uruchamiaj
testu ponownie, żeby „trafić lepiej”.

## 6. Raport dla użytkownika

Na koniec podaj: warianty i ich wyniki na dev i OOD, wybrany wariant, jednorazowy
wynik na `essays_2021`, czas treningu i GPU, nazwę gałęzi oraz to, czego nie
udało się zrobić.
