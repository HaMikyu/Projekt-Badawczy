# Wyniki ML — detekcja tekstu AI (eseje)

## Werdykt

Lokalny **System One option scorer** (DistilBERT, kontrakt Jev: `state + options -> wybór`)
na mieszance esejów bije stary DistilBERT v3 na Ghostbuster OOD i mocno obcina fałszywe AI.

| model | OOD acc | human TNR | AI TPR |
|---|---|---|---|
| DistilBERT v3 (stary, Kaggle+HC3+Pile) | 0,838 | 0,705 | 0,970 |
| System One v1 (same essays, bez GB train) | 0,503 | 0,007 | 0,999 |
| System One v2 (esseje + GB train) | 0,931 | 0,863 | 1,000 |
| System One v3 | 0,946 | 0,900 | 0,993 |
| System One v4 (DistilBERT, 25.11.2026) | 0,985 | 0,993 | 0,977 |
| **System One v5 (DeBERTa-v3, final od 26.11.2026)** | **0,995** | **0,993** | **0,997** |

v4 to v3 douczony na +23 tys. esejów IvyPanda (ludzkie + AI) we fragmentach
jak w aplikacji — szczegóły w sekcji [System One v4](#system-one-v4-douczenie-na-gpu-25112026).
v5 to nowy enkoder (DeBERTa-v3-base) trenowany na danych v4 + nowych esejach AI
(GPT-4 Turbo, GPT-4o, Llama 3 70B, Command R+, LLMEssay-100K, Llamy naśladujące uczniów) —
sekcja [System One v5](#system-one-v5-deberta-v3-i-wiecej-generatorow-26112026).
Na esejach akademickich v5 wykrywa znacznie więcej AI przy 0–0,7% fałszywych alarmów;
poza esejami (Wikipedia, przemówienia, krótkie teksty osób nieanglojęzycznych) myli się częściej niż v4.

## Co to jest System One

Nie chatbot. Dla eseju scoruje dwie opcje:

- `human` (kryteria: nierówny rytm, szczegóły, głos studenta)
- `ai` (kryteria: szablon, wygładzony styl, generyczna teza)

Softmax po score'ach = skalibrowane `p_ai` (+ temperature scaling).

## Dane (essay-first)

Trening: Kaggle essays + DAIGT v3 + ArguGPT + Ghostbuster **train**
(humans z GB x10) + formalne human z Pile jako twarde negatywy.
OOD: Ghostbuster **test** (nigdy w treningu), length-matched.

| split | n |
|---|---|
| train | 52 526 |
| val / test | 6 566 |
| OOD GB test | 598 |

## Wyniki System One v3

### Test IID

| metryka | wartość |
|---|---|
| accuracy | **0,994** |
| precision | 0,989 |
| recall | 0,998 |
| F1 | **0,994** |
| TNR (human) | 0,989 |

### OOD Ghostbuster test

| metryka | wartość |
|---|---|
| accuracy | **0,946** |
| precision | 0,908 |
| recall | 0,993 |
| F1 | **0,949** |
| TNR (human) | **0,900** |
| threshold | 0,83 |
| temperature | 0,74 |

Macierz OOD (`[[TN, FP], [FN, TP]]`):

```text
[[269, 30],
 [  2, 297]]
```

Czas treningu v3: ~37 min (RTX 3070 Ti, batch 12, max_length 384, 4 epoki).

## Odpalenie

```powershell
.\.venv\Scripts\python.exe train\build_data.py
.\.venv\Scripts\python.exe train\train_system_one.py
.\.venv\Scripts\python.exe train\export_system_one.py
.\.venv\Scripts\python.exe train\predict_system_one.py naturalny_essay.txt
```

Artefakty:

- wagi: `models/system-one-final/`
- metryki: `train/runs/system-one-v3/metrics.json`
- dane: `data/prepared/` (gitignore)

Opcjonalnie prawdziwy Jev API: `train/jev_client.py` (wymaga `TYPESAFE_API_KEY`).

## Aplikacja: ocena fragmentami + finalny test na esejach spoza treningu

System One widzi tylko 384 tokeny, czyli zwykle sam wstęp eseju. Aplikacja
(`app/helpers.py`) dzieli więc tekst na fragmenty ~320 tokenów (bez cięcia zdań),
ocenia każdy osobno i uśrednia `p_ai`:
**AI ≥ 0,5 · NIEPEWNE 0,3–0,5 · CZŁOWIEK < 0,3**.

Ustawienia dobrano na zbiorze dev (`train/build_dev_set.py`: 290 ludzkich esejów
IvyPanda + 282 eseje AI na te same tematy z ChatGPT 4.1 / DeepSeek / Llama 3 / Mistral
+ 16 esejów od Claude):

| wariant (dev) | ludzkie → AI | AI wykryte |
|---|---|---|
| tylko pierwszy fragment | 10,0% | 87,6% |
| **średnia z fragmentów, próg 0,5** | **3,4%** | 84,2% |
| średnia z fragmentów, próg 0,7 | 2,4% | 77,2% |

Finalny test, uruchomiony raz po zamrożeniu ustawień: `essays_2021/`, 300 ludzkich
esejów StudyCorgi z 2021 r. (zbiór nigdy nie używany do treningu ani strojenia),
skrypt `train/final_test_essays_2021.py`:

| detektor | człowiek | niepewne | AI (fałszywy alarm) |
|---|---|---|---|
| poprzednia aplikacja (średnia DistilBERT baseline + statystyczny) | – | – | 93,3% |
| **System One, fragmenty** | **80,3%** | 12,7% | **7,0%** |

Plan douczenia modelu na GPU, który ma zmniejszyć te 7%: [`GPU_TRAINING.md`](./GPU_TRAINING.md).
Wykonany — patrz niżej.

## System One v4: douczenie na GPU (25.11.2026)

Wykonane wg [`GPU_TRAINING.md`](./GPU_TRAINING.md), gałąź `system-one-v4`,
RTX 3070 Ti 8 GB, start z wag v3.

### Dane (`train/build_data_v4.py`)

- v3 (bez zmian w podziale) + **11 644 ludzkich esejów IvyPanda i 11 644 esejów AI**
  z tego samego zbioru LLM (GPT 2951, DeepSeek 2938, Llama 2882, Mistral 2873),
  ludzkie dobierane do długości AI w koszykach długości, podział po tytułach.
- Wykluczone: tytuły i teksty zbioru dev, tytuły `essays_2021` oraz wszystko, co
  ma > 20 wspólnych 8-gramów z dev, `essays_2021` lub Ghostbusterem
  (**ludzkie eseje Ghostbustera pochodzą z IvyPanda** — 873 teksty usunięte,
  żeby OOD zostało OOD).
- Wariant „okna”: z każdego eseju 1–3 fragmenty ~320 tokenów tym samym
  algorytmem co `app/helpers.py` (początek + losowy środek + koniec).
- Rozmiar: 105 830 / 13 192 / 13 019 (okna), 71 244 / 8 898 / 8 797 (pełne teksty).

Uwaga: przy budowie znalazłem bliskie duplikaty dev w puli v3
(`pile_formal_human`, maks. 613 wspólnych 8-gramów) — usunięte z train/val v4.
Wynik v3 na dev mógł więc być lekko zawyżony.

### Warianty (wybór tylko na dev + Ghostbuster OOD)

| wariant | dane | lr | epoki | czas | dev ludzkie→AI @0,5 | dev AI wykryte @0,5 | OOD acc | OOD TNR | OOD TPR |
|---|---|---|---|---|---|---|---|---|---|
| v3 (punkt odniesienia) | — | — | — | — | 2,71% | 84,2% | 0,945 | 0,896 | 0,993 |
| a | pełne teksty | 1e-5 | 2 | 23 min | 0,0% | 97,5% | **0,987** | 0,983 | 0,990 |
| **b (wybrany)** | **okna** | **1e-5** | 2 | 35 min | **0,0%** | **100%** | 0,985 | **0,993** | 0,977 |
| c | okna | 2e-5 | 2 | 35 min | 0,0% | 100% | 0,980 | 0,993 | 0,967 |

(v3 na dev policzony na obecnej wersji `data/dev`: 295 ludzkich + 279 AI.)

Wszystkie trzy spełniają kryteria akceptacji. Wybrany **b**: najmniej fałszywych
alarmów na ludzkich tekstach OOD (2/299 vs 5/299 dla a) przy perfekcyjnym dev
nawet przy progu 0,7 (0% / 99,6%). Każdy generator na dev: 100% (v3: Mistral 72,9%,
Llama 74,1%). Macierz OOD b: `[[297, 2], [7, 292]]`. IID (okna): acc 0,994, F1 0,994.

### Finalny test `essays_2021` (uruchomiony raz, po wyborze wariantu)

| detektor | człowiek | niepewne | AI (fałszywy alarm) |
|---|---|---|---|
| System One v3, fragmenty | 80,3% | 12,7% | 7,0% |
| **System One v4, fragmenty** | **99,3%** | 0,7% | **0,0%** |

v3 przeliczony na tej samej lokalnej kopii esejów daje identyczne liczby co
pierwotny test, więc porównanie jest bezpośrednie. Dwa eseje v4 ocenił jako
NIEPEWNE (średnie p_ai 0,48 i 0,49), mediana p_ai wynosi 0,0005.

**Ograniczenia, których ten wynik nie znosi:**

- `essays_2021` zawiera wyłącznie teksty ludzkie, więc mierzy tylko fałszywe alarmy.
  Wykrywalność AI pochodzi z dev i Ghostbustera.
- AI w dev pochodzi z tej samej rodziny danych LLM co nowy trening (te same
  generatory, inne tematy), więc 100% na dev jest pomiarem bliskim domenie.
  Bardziej miarodajne jest Ghostbuster OOD: 97,7% AI wykryte (v3: 99,3%, czyli lekki spadek).
- StudyCorgi i IvyPanda to podobne serwisy z esejami. Model nauczył się tego stylu
  jako ludzkiego, więc na innym typie ludzkiego tekstu wynik może być gorszy.

### Test zewnętrzny na gotowych benchmarkach (nowe generatory, różne gatunki)

`train/eval_external.py`, uruchomiony raz, bez strojenia progów. Wyniki w `train/external_test_results.json`.

- **WasItAIGenerated Benchmark 2026-09** (HF `wasitaigeneratedcom/ai-detector-benchmark-2026-09`).
  Zawiera teksty AI z ok. 40 modeli z lat 2025–2026 (GPT-5/6, Claude 4/5, Gemini 3, DeepSeek V4, Qwen, Grok…),
  parafrazy, teksty po 6 humanizerach komercyjnych i teksty ludzkie (mowy, dokumentacje,
  teksty uczniów, artykuły) oraz werdykty GPTZero / Pangram / ZeroGPT na tych samych tekstach.
- **TextSight 2026** (HF `textsightai/ai-text-detection-benchmark-2026`): ludzkie teksty z Wikipedii sprzed 2022,
  z PMC (autorzy spoza krajów anglojęzycznych) i z literatury oraz 56 tekstów AI (akademickie i encyklopedyczne).
- Tylko język angielski, teksty od 80 słów. Usunięto 101 tekstów, które mają ponad 20 wspólnych 8-gramów z treningiem v4
  (głównie eseje PERSUADE, które już były w danych Kaggle). Zostało 2066 tekstów.

| grupa | n | v3 → AI | **v4 → AI** | v4 → niepewne |
|---|---|---|---|---|
| WasItAI: teksty ludzkie (fałszywy alarm) | 256 | 17,6% | **1,9%** | 0,4% |
| TextSight: teksty ludzkie (fałszywy alarm) | 635 | 20,5% | **8,5%** | 1,3% |
| – w tym Wikipedia | 155 | 63% | 30% | 2% |
| WasItAI: surowe AI (wykryte) | 671 | 44,7% | **35,6%** | 3,7% |
| WasItAI: AI parafrazowane | 44 | 56,8% | 45,5% | 0% |
| WasItAI: AI po humanizerze | 404 | 33,9% | 8,7% | 9,9% |
| TextSight: AI akademickie | 29 | 7% | **69%** | 14% |
| TextSight: AI encyklopedyczne | 27 | 100% | 100% | 0% |

AUC (ludzkie vs surowe AI, liczone z p_ai):

| zbiór | v3 | v4 |
|---|---|---|
| WasItAI | 0,654 | 0,743 |
| TextSight | 0,633 | 0,960 |

Te same teksty ocenione przez detektory komercyjne (dane benchmarku):

| podzbiór | GPTZero | Pangram | v4 |
|---|---|---|---|
| surowe AI, podzbiór GPTZero (n=188) | 60% | – | 36% |
| surowe AI, podzbiór Pangram (n=143) | – | 42% | 36% |
| po humanizerze, podzbiór GPTZero (n=278) | 24% | – | 9% |
| ludzkie, podzbiór GPTZero/Pangram (n≈42) | 0% | 0% | 7% |

**Wniosek.** v4 rzadko fałszywie oskarża ludzi także poza esejami, ale słabo wykrywa
teksty AI z nowych modeli w innych gatunkach niż esej, np. abstrakty, newsy, recenzje i przepisy.
Wykrywa ich 36%, mniej niż v3 (45%) i GPTZero (60%). v4 lepiej porządkuje teksty (AUC wyższe niż v3),
ale próg 0,5 dobrany na esejach jest poza tą domeną bardzo ostrożny. Progu nie zmieniamy pod ten test.
Humanizery skutecznie ukrywają AI przed v4 (9%). Teksty AI z benchmarku nie są esejami,
więc test nie odpowiada wprost na pytanie o eseje z nowych modeli; to nadal brakujący pomiar.

Odtworzenie:

```powershell
.\.venv\Scripts\python.exe train\build_data_v4.py
bash train/run_v4_variants.sh a b c     # trening + eval_dev, bez finalnego testu
$env:DETEKTOR_SYSTEM_ONE_DIR="models\system-one-v4"; .\.venv\Scripts\python.exe train\final_test_essays_2021.py
```

Metryki: `train/runs/system-one-v4{a,b,c}/` (gitignore), wagi: `models/system-one-v4/`
(do 26.11.2026 = `models/system-one-final/`).

## System One v5: DeBERTa-v3 i więcej generatorów (26.11.2026)

Cel: lepsza wykrywalność esejów AI z generatorów, których v4 nie widział, bez wzrostu
fałszywych alarmów. Zakres: tylko eseje. Gałąź `system-one-v4`, RTX 3070 Ti 8 GB.

### Dane (`train/build_data_v5.py`)

Baza: pełne dane v4 (`data/prepared_v4_full`) + nowe zbiory esejów (tylko angielskie):

| źródło | co to jest | train / val / test |
|---|---|---|
| artnitolog (HF `artnitolog/llm-generated-texts`, część „essay”) | eseje GPT-4 Turbo, GPT-4o, Llama 3 70B, Command R+ na 1000 tematach IvyPanda | 2733 / 584 / 581 |
| ta sama kolekcja, **rodziny wyłączone** | Claude 3 Opus, YandexGPT 3 Pro, GigaChat Pro — tylko w teście, na tematach testowych | 0 / 0 / 420 |
| LLMEssay-100K (HF `chiikabu-labs/LLMEssay-100K`, 3 z 10 shardów) | eseje AI po usunięciu markdownu | 6000 / 750 / 1000 |
| DAIGT v4 (HF `ramensoft/daigt_v4`) | Llamy dostrojone do stylu uczniów (PERSUADE) | 6233 / 794 / 570 |
| IvyPanda (eseje ludzkie nieużyte w v4) | ludzkie eseje dobrane długością | 12 474 / 1564 / 1497 |

- Podział artnitolog po tematach 70/15/15; rodziny wyłączone pojawiają się wyłącznie na tematach testowych.
- Te same zabezpieczenia co w v4: wykluczenie wszystkiego, co ma > 20 wspólnych 8-gramów z dev, `essays_2021`,
  Ghostbusterem (train i test) i benchmarkami zewnętrznymi (WasItAI, TextSight); test sprawdzany dodatkowo
  względem całego train+val. Kontrola przecieków po budowie: 0 / 0.
- Usunięte przy budowie: 987 ludzkich esejów artnitolog (to są eseje Ghostbustera), 314 nowych esejów IvyPanda
  pokrywających się z Ghostbusterem, 222 teksty testowe DAIGT pokrywające się z treningiem.
- Rozmiar: 98 690 pełnych tekstów w train (51% AI); okna jak w aplikacji: 151 312 / 19 422 / 19 391.

**Zastrzeżenie.** Ludzkie eseje artnitolog to eseje Ghostbustera (te same tematy IvyPanda), więc Ghostbuster OOD ma teraz
pokrycie tematyczne (nie tekstowe) z nowymi esejami AI w treningu. LLMEssay i DAIGT v4 są
w rodzinie treningowej v5, więc ich testy mierzą generalizację na nowe teksty, nie na nowe generatory.
Na nowe generatory odpowiada test rodzin wyłączonych i test zewnętrzny.

### Warianty (wybór tylko na walidacji, dev i Ghostbuster OOD)

| wariant | enkoder | start | lr | epoki | czas | val bal. acc | dev ludzkie→AI @0,5 | dev AI wykryte | OOD acc | OOD TNR | OOD TPR |
|---|---|---|---|---|---|---|---|---|---|---|---|
| v4 (punkt odniesienia) | DistilBERT | — | — | — | — | — | 0,0% | 100% | 0,985 | 0,993 | 0,977 |
| a | DistilBERT | wagi v4 | 1e-5 | 2 | 53 min | 0,980 | 0,34% | 100% | 0,983 | 0,967 | 1,000 |
| **d (wybrany)** | **DeBERTa-v3-base** | wstępnie wytrenowany | 2e-5 | 1 | 9 h 3 min | **0,990** | **0,0%** | **100%** | **0,995** | **0,993** | **0,997** |

Macierz OOD d: `[[297, 2], [1, 298]]`. IID (okna, próg z walidacji 0,85): acc 0,990, TNR 0,985, TPR 0,994.

**Błąd naprawiony po drodze.** Pierwszy trening DeBERTy dawał `loss = nan` od pierwszego kroku
(5 h GPU zmarnowane). transformers 5 wczytuje wagi w typie z checkpointu, a DeBERTa-v3 jest zapisana
w fp16; AdamW na wagach fp16 (eps 1e-8 = 0 w fp16) psuje je po jednym kroku. Poprawka w
`train/system_one_model.py`: enkoder zawsze w fp32 (`dtype=torch.float32`), obliczenia w bf16 (AMP).
`train/train_system_one_v5.py` przerywa trening, gdy loss stanie się NaN.

### Testy esejowe kodem aplikacji (`train/eval_essay_suite.py`)

Fragmenty ~320 tokenów, średnia p_ai, progi aplikacji 0,5 / 0,3; zestaw powstał przed treningiem,
nie był używany do wyboru wariantu.

| test | n | v4 | v5a | **v5 (d)** |
|---|---|---|---|---|
| rodziny wyłączone (Claude 3 Opus, YandexGPT, GigaChat): AI wykryte | 420 | 99,5% | 100% | **100%** |
| rodziny artnitolog z treningu, tematy testowe: AI wykryte | 581 | 100% | 100% | **100%** |
| LLMEssay-100K: AI wykryte | 1000 | 40,8% | 100% | **100%** |
| DAIGT v4 (Llamy w stylu uczniów): AI wykryte | 570 | 50,9% | 88,4% | **94,2%** |
| ludzkie eseje IvyPanda do LLMEssay: fałszywy alarm | 1497 | 0,27% | 0,80% | 0,53% |
| ludzkie eseje IvyPanda do LLMEssay: niepewne | 1497 | 0,13% | 1,27% | 0,27% |
| dev: ludzkie → AI / AI wykryte | 295 / 279 | 0% / 100% | 0,34% / 100% | **0% / 100%** |

Rodziny wyłączone v4 też wykrywał prawie w całości, więc ten test okazał się za łatwy, by rozróżnić modele.
Największy zysk to długie eseje LLMEssay (v4 był na nich źle skalibrowany: AUC 0,99, ale 39% oceniał jako ludzkie)
i Llamy dostrojone do stylu uczniów.

### Finalny test `essays_2021` (uruchomiony raz, po wyborze wariantu)

| detektor | człowiek | niepewne | AI (fałszywy alarm) |
|---|---|---|---|
| System One v4, fragmenty | 99,3% | 0,7% | 0,0% |
| **System One v5, fragmenty** | **98,7%** | 0,7% | 0,7% |

v5 ocenił 2 z 300 ludzkich esejów jako AI (v4: 0). To mieści się w kryterium akceptacji (≤ 2%),
ale jest drobnym pogorszeniem. **Uwaga:** `essays_2021` był już raz użyty dla v4, więc jest to
jego drugie użycie dla nowego kandydata; wybór wariantu nastąpił wcześniej i bez niego.

### Wagi i aplikacja

- `models/system-one-v5/` = `models/system-one-final/`: wagi w fp16 (368 MB zamiast 735 MB),
  przy wczytaniu rzutowane do fp32; różnica p_ai fp16 vs fp32 na 80 tekstach dev ≤ 0,01 p.p.
- Przy modelu zapisana jest konfiguracja enkodera (`config.json`), a `train/predict_system_one.py`
  buduje enkoder z konfiguracji (`pretrained=False`), więc aplikacja nie pobiera wag bazowych z Hugging Face.
- DeBERTa-v3-base ma ok. 184 mln parametrów (DistilBERT: 66 mln), więc ocena jest wolniejsza, zwłaszcza na CPU. v4 zostaje w `models/system-one-v4/`
  (`DETEKTOR_SYSTEM_ONE_DIR=models\system-one-v4` przywraca go w aplikacji).

### Test zewnętrzny (te same benchmarki co dla v4, poza esejami)

`train/eval_external.py v3=… v4=… v5=…`, uruchomiony raz po wyborze modelu, bez strojenia progów.
Filtr przecieków obejmuje teraz trening v4 i v5. Usunięte teksty są te same co wcześniej (101), zostało 2066.
Wyniki: `train/external_test_results.json`.

| grupa | n | v3 → AI | v4 → AI | **v5 → AI** |
|---|---|---|---|---|
| WasItAI: teksty ludzkie (fałszywy alarm) | 256 | 17,6% | **1,9%** | 16,8% |
| – w tym teksty uczniów (ELLIPSE + chińscy uczniowie CSEE) | 53 | 22,6% | 1,9% | 30,2% |
| – w tym przemówienia | 70 | 30,0% | 0,0% | 25,7% |
| TextSight: teksty ludzkie (fałszywy alarm) | 635 | 20,5% | **8,5%** | 22,2% |
| – w tym Wikipedia | 155 | 63% | 30% | 68% |
| – w tym literatura / PMC | 297 / 183 | 5,4% / 8,7% | 0,3% / 3,3% | 7,7% / 7,1% |
| WasItAI: surowe AI (wykryte) | 671 | 44,7% | 35,6% | **61,0%** |
| WasItAI: AI parafrazowane | 44 | 56,8% | 45,5% | **72,7%** |
| WasItAI: AI po humanizerze | 404 | 33,9% | 8,7% | **44,1%** |
| TextSight: AI akademickie | 29 | 7% | 69% | **100%** |

AUC (ludzkie vs surowe AI): WasItAI 0,654 / 0,743 / 0,744, TextSight 0,633 / 0,960 / 0,977 (v3 / v4 / v5).

Te same podzbiory oceniane przez detektory komercyjne:

| podzbiór | GPTZero | Pangram | v4 | v5 |
|---|---|---|---|---|
| surowe AI, podzbiór GPTZero (n=188) | 60% | – | 36% | **65%** |
| surowe AI, podzbiór Pangram (n=143) | – | 42% | 36% | **65%** |
| po humanizerze, podzbiór GPTZero (n=278) | 24% | – | 9% | **40%** |
| ludzkie, podzbiór GPTZero (n=42) | **0%** | – | 7% | 31% |

**Wniosek (bez upiększania).** v5 wykrywa teksty AI z ok. 40 nowych modeli także poza esejami
lepiej od v4 i od GPTZero/Pangram na tych samych tekstach, również po humanizerach.
Ceną jest znacznie więcej fałszywych alarmów na ludzkich tekstach **spoza esejów akademickich**
(Wikipedia, przemówienia, krótkie teksty uczniów). AUC na WasItAI się nie zmieniło, więc v5 nie jest tu lepszym
klasyfikatorem, tylko przesuniętym w stronę „AI”. Na esejach akademickich (dev, `essays_2021`, IvyPanda) fałszywe alarmy
v5 nadal wynoszą 0–0,7%.

**Znana stronniczość v5: krótkie, szablonowe teksty osób uczących się angielskiego.** Fałszywe alarmy wśród uczniów
dotyczą prawie wyłącznie korpusu CSEE (chińscy uczniowie, listy egzaminacyjne ok. 130 słów pisane według szablonu
„Dear Jim… Firstly… What's more… Last but not least”). v5 oznaczył jako AI 16 z 38 takich tekstów, często z p_ai ≈ 0,99
(v4: 0 z 38). Na ELLIPSE (amerykańscy uczniowie uczący się angielskiego, dłuższe eseje) v5 ma 0%.
Ta sama słabość dotyczy innych detektorów: szablonowy styl osób piszących w obcym języku przypomina tekst AI.
**Wyniku v5 nie należy używać jako dowodu wobec krótkich (< 200 słów) tekstów osób nieanglojęzycznych.**
Progu nie zmieniamy pod ten test. Naprawa to zadanie na v6: ludzkie teksty uczniów i osób nieanglojęzycznych
w treningu (np. ICNALE, ELLIPSE), a nie dopasowanie do CSEE.

Odtworzenie v5:

```powershell
.\.venv\Scripts\python.exe train\build_data_v5.py
bash train/run_v5.sh a d                                     # trening + eval_dev
.\.venv\Scripts\python.exe train\eval_essay_suite.py v4=models/system-one-v4 v5d=train/runs/system-one-v5d/export
.\.venv\Scripts\python.exe train\final_test_essays_2021.py   # raz, po wyborze (models/system-one-final = v5)
.\.venv\Scripts\python.exe train\eval_external.py v3=train/runs/system-one-v3/export v4=models/system-one-v4 v5=models/system-one-v5
```

Metryki: `train/runs/system-one-v5{a,d}/` i `train/runs/essay_suite/` (gitignore).
