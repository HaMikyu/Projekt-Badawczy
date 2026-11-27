# Plan: prawdziwy System One (open source) do detekcji esejów AI

Stan na 26.11.2026. Sprzęt: RTX 3070 Ti 8 GB, 31 GB RAM, ok. 106 GB wolnego dysku, Windows 10.

## 1. Co mamy teraz, a czym jest System One

**Nasz model (v1–v5)** to klasyfikator w stylu „option scorer”: enkoder (DistilBERT, potem DeBERTa)
czyta esej razem z opisem opcji *human* / *ai* i zwraca dwie liczby. Jest jedno, zaszyte na stałe pytanie,
okno 384 tokenów (stąd fragmenty w aplikacji) i brak typowanego API.

**System One** (pierwowzór: Jev firmy TypeSafe, model zamknięty, API `POST /v1/systemone`) to model
decyzyjny, a nie klasyfikator jednego zadania:

- wejście: `state` (tekst / JSON) + dowolne `questions` pisane w języku naturalnym,
  każde z typem: **Choice** (wybór z opcji), **Score** (skala uporządkowana), **Noul** (tak/nie → P(tak));
- jedno przejście modelu, bez generowania tekstu; wszystkie pytania są liczone na tym samym stanie;
- wyjście: skalibrowane prawdopodobieństwa (trening proper scoring rules + kalibracja temperaturą).

Otwarte odpowiedniki (stan z README, nie weryfikowane przez nas):

| projekt | baza | odczyt decyzji | licencja | `/v1/systemone` | uwagi |
|---|---|---|---|---|---|
| **Kev** (`jaredpalmer/kev`) | Qwen3.5 0.8B / 4B / 9B + LoRA r16, 27B full FT | pointer head: `</opt>` vs `<decide>` | Apache-2.0 | tak | najbliższa rekonstrukcja Jev, wagi na HF, kontekst 8192 |
| Nimble (`bespokelabsai/nimble`) | Qwen3.5-9B + LoRA, 27B | logity tokenów-kodów odpowiedzi | nie znaleziono | nie | jawny przepis na dane (2676 przykładów) |
| Laya (`NandhaKishorM/laya`) | ModernBERT-large 421M | markery opcji w enkoderze | Apache-2.0 (część NC) | tak | najbliżej tego, co mamy teraz |
| Von (`wfzyx/von`) | ModernBERT 395M | OptionMarker | Apache-2.0 | tak | **zarchiwizowany 19.10.2026** |
| system-one-open | Gemma 4 E2B / Gemma 3 270M | logity slotów | MIT | nie | wag jeszcze nie ma na HF |

**Wniosek:** Laya i Von to w zasadzie nasze podejście (enkoder + opcje). „Prawdziwy” System One to
**dekoderowy LLM z głowicą decyzyjną i typowanym kontraktem** — wzorem jest Kev.

## 2. Decyzja architektoniczna

**Startujemy z wag Kev (Apache-2.0) i douczamy je do esejów, zamiast budować System One od zera.**

- Dostajemy gotowy kontrakt (Choice/Score/Noul, `/v1/systemone`, pointer head, kalibracja) i ogólną
  umiejętność czytania kryteriów. Nasza praca to dane i trening w domenie esejów.
- Cały esej mieści się w jednym przejściu (kontekst 8192, esej ≈ 800–1500 tokenów), więc znika
  problem „model widzi tylko wstęp” i dzielenie na fragmenty staje się opcją, a nie koniecznością.
- Punkt odniesienia do pokonania: **System One v5 (DeBERTa)**. Fallback: v5 zostaje w aplikacji.

### Czy się zmieści na 8 GB — szacunek (do potwierdzenia w fazie 0)

| model | wagi | tryb treningu | VRAM trening (seq 2k, batch 1, gradient checkpointing) | 8 GB? | VRAM inferencja |
|---|---|---|---|---|---|
| **Kev-0.8B** | 1,6 GB bf16 | LoRA bf16 | ok. 4–5 GB (README Kev: trening mieści się w 4 GB) | **tak, z zapasem** | ok. 2 GB |
| **Kev-4B** | 8 GB bf16 → ok. 2,6 GB NF4 | QLoRA (4-bit + LoRA) | ok. 6–7 GB | **na granicy, seq ≤ 2048** | ok. 3,5 GB (4-bit) |
| Kev-9B | ok. 5,5 GB NF4 | QLoRA | > 8 GB | **nie** | ok. 6,5 GB, bardzo ciasno |
| Kev-27B | 51 GB bf16 | — | — | nie | nie |

Kluczowe sztuczki pamięciowe: liczyć głowicę tylko na pozycjach `</opt>`/`<decide>` (nie pełne logity
słownika 150 tys. × długość), gradient checkpointing, batch 1 + akumulacja, 8-bitowy AdamW
(paged), stan eseju liczony raz na wszystkie pytania.

**Prawdziwe ograniczenie to czas, nie pamięć.** Kev-4B: ok. 12,5 tys. przykładów × 2 epoki = 1 h na H100.
3070 Ti jest ok. 10–20× wolniejszy, a 4-bit dodaje narzut, więc szacunkowo:

| model | przykładów (~1k tokenów) na noc (10 h) | uwagi |
|---|---|---|
| Kev-0.8B, LoRA | ok. 60–100 tys. | do iteracji nad przepisem i danymi |
| Kev-4B, QLoRA | ok. 10–15 tys. | model finalny, 3–5 nocy |

Dlatego plan: **duża pula danych (setki tysięcy esejów)**, ale do treningu trafia
**starannie wybrany podzbiór** (różnorodność + trudne przykłady), a nie wszystko jak leci.

## 3. Fazy pracy

### Faza 0 — rozpoznanie i środowisko (1–2 dni)

1. **Środowisko: WSL2 + Ubuntu + CUDA.** Qwen3.5 ma warstwy Gated DeltaNet, których jądra
   (flash-linear-attention) wymagają Tritona; bitsandbytes i vLLM też działają pewnie tylko pod Linuksem.
   Natywny Windows traktujemy jako plan B (triton-windows), sprawdzany w 30 minut, nie dłużej.
2. Pobrać `jaredpalmer/kev-0.8b` i `kev-4b`, sprawdzić licencje i model cardy, uruchomić serwer Kev.
3. **Pomiar, nie szacunek:** VRAM i tokeny/s dla 0.8B (bf16) i 4B (NF4) przy seq 1024 / 2048 / 4096,
   inferencja i krok treningu LoRA. Wynik decyduje o wielkości modelu finalnego.
4. **Zero-shot baseline:** Kev-0.8B i Kev-4B bez douczania na naszym dev i teście esejowym,
   pytanie Noul „czy ten esej napisał model AI?” z kryteriami jak w `jev_client.py`.
5. **Decyzja go/no-go:** jeśli QLoRA 4B nie mieści się przy seq 2048 lub działa < ok. 300 tok/s,
   model finalny = Kev-0.8B (i ewentualnie zwiększamy dane zamiast modelu).

### Faza 1 — duży zbiór danych (3–5 dni, w tym generowanie nocami)

**Zasada nadrzędna: ludzkie teksty tylko sprzed 2022 r.** (gwarancja braku AI) i każde źródło z kartą
danych (licencja, liczba, okres, styl, kto pisał).

Ludzkie eseje (twarde negatywy):

| źródło | ok. liczba | po co | licencja (do sprawdzenia) |
|---|---|---|---|
| IvyPanda (mamy) | 40 tys.+ | eseje akademickie | do sprawdzenia |
| PERSUADE 2.0 | 25 tys. | uczniowie 6–12 klasa (USA) | CC BY-NC-SA |
| ELLIPSE | 6,5 tys. | **osoby uczące się angielskiego** | NC |
| ICNALE (Written Essays) | 5,6 tys. | **uczniowie z Azji, krótkie eseje 200–300 słów** — wprost na lukę v5 | wymaga wniosku |
| ASAP / ASAP++ | 17 tys. | krótkie eseje szkolne | do sprawdzenia |
| BAWE, MICUSP | ok. 3,6 tys. | eseje studenckie UK / USA | Oxford Text Archive / akademicka |
| Kaggle essays, Pile formal (mamy) | dziesiątki tys. | różnorodność | jak w v3 |

Eseje AI:

| źródło | ok. liczba | generatory |
|---|---|---|
| LLMEssay-100K (całość, 10 shardów) | 100 tys. | modele 2026 |
| DAIGT v2 / v3 / v4 | 50 tys.+ | GPT-3.5/4, Llama, Mistral, Claude, PaLM, Llamy dostrojone |
| artnitolog, ArguGPT, IvyPanda-LLM (mamy) | 30 tys.+ | GPT-4/4o, Llama 3, Command R+, DeepSeek, Mistral |
| MAGE, M4GT (części esejowe) | dziesiątki tys. | dziesiątki modeli |
| **własne generowanie lokalne** (llama.cpp, Q4) | 2–3 tys. na noc na model | Qwen3.5, Gemma 3/4, Llama 3.x, Mistral Small, Phi-4, DeepSeek-distill |

Własne generowanie dokłada to, czego brakuje w gotowych zbiorach:
- eseje na **te same tematy co ludzkie** (PERSUADE, IvyPanda, ELLIPSE), żeby model nie uczył się tematu;
- warianty stylu: „jak uczeń 10. klasy”, „jak osoba ucząca się angielskiego”, krótkie listy egzaminacyjne,
  literówki — czyli **trudne pozytywy** podobne do ludzkich negatywów;
- **AI-edycja** (ludzki esej „popraw gramatykę / styl” przez LLM) i **parafraza/humanizacja** AI-eseju —
  osobne klasy, bo to dziś główny sposób obchodzenia detektorów.

Czyszczenie i kontrola przecieków (rozszerzenie tego, co już działa w v4/v5):
- normalizacja (usuwanie markdownu, nagłówków „Title:”, „As an AI…”, jednolite cudzysłowy i spacje) —
  inaczej model uczy się formatowania zamiast stylu;
- dopasowanie długości human/AI w koszykach; deduplikacja MinHash + 8-gramy (`ShingleIndex`);
- wykluczenie wszystkiego, co pokrywa się z: dev, `essays_2021`, Ghostbusterem, WasItAI, TextSight,
  nowym zbiorem finalnym (faza 4); podział **po tematach i po rodzinach generatorów**, nie po tekstach.

Docelowo: pula ok. 300–500 tys. tekstów, z czego do treningu 4B trafia 40–60 tys. przykładów
wybranych przez **hard mining**: v5 (szybki) i Kev-0.8B oceniają całą pulę, bierzemy przykłady błędne,
niepewne i z rzadkich źródeł, plus losową próbkę dla pokrycia.

### Faza 2 — format System One (1–2 dni)

Każdy przykład to żądanie w kontrakcie `/v1/systemone`: `state = {"genre": "student_essay", "text": …}`
i kilka pytań na ten sam stan:

| id pytania | typ | etykieta skąd |
|---|---|---|
| `ai_generated` | Noul | wszystkie źródła (główne pytanie) |
| `origin` | Choice: human / ai / ai_edited / humanized | źródło + własne generowanie |
| `ai_share` | Score: 0–4 (udział AI) | eseje mieszane (sklejane fragmenty human + AI) |
| `model_family` | Choice: gpt / claude / llama / qwen / gemini / mistral / deepseek / other / none | tylko gdzie znana; pytanie pomocnicze |

- Treść `instructions` i `criteria` **losowana z puli parafraz** (i po polsku/angielsku), żeby model
  uczył się czytać pytanie, a nie zapamiętał jednego promptu — to jest cała wartość System One.
- **Replay ogólnych decyzji Kev (ok. 15–20% batchy)**, żeby douczenie nie zniszczyło ogólnej umiejętności
  (sprawdzane na `jaredpalmer/kev-suites`).
- Nie dodajemy pytań o cechy autora (np. „czy pisze osoba nieanglojęzyczna”) — używamy ich tylko
  do oceny sprawiedliwości.

### Faza 3 — trening (najpierw 0.8B, potem 4B)

1. **Kev-0.8B, LoRA bf16** — szybkie iteracje (1 noc = pełny przebieg): dobór mieszanki danych,
   lr, długości stanu (1024 vs 2048 vs pełny esej), wagi Brier.
2. **Kev-4B, QLoRA** — 2–3 noce na finalnym przepisie (jeśli faza 0 przejdzie).
3. Strata: cross-entropy na pointer head **+ Brier** (proper scoring rule, jak RLCD/Laya); potem
   temperatura per typ pytania i liczba opcji, dopasowana na walidacji.
4. Higiena treningu z lekcji v5: sprawdzić dtype wag przed długim biegiem (fp16 → NaN), przerywać
   przy NaN, checkpoint co N kroków, log postępu do pliku, wybór tylko na walidacji/dev.

### Faza 4 — ocena (protokół zamrożony przed treningiem)

- **Nowy zbiór finalny** (raz, na końcu): ludzkie eseje sprzed 2022 z niewykorzystanego źródła +
  eseje AI od rodzin wyłączonych z treningu (np. całe rodziny Gemma i Phi + Claude 3 Opus / YandexGPT /
  GigaChat z artnitolog). `essays_2021` był użyty dwa razy, więc zostaje tylko jako test regresji.
- **Wycinki sprawiedliwości:** osoby uczące się angielskiego (ELLIPSE test, ICNALE test), teksty < 200 słów,
  CSEE z WasItAI (tylko ewaluacja, nigdy trening).
- **Test zewnętrzny:** WasItAI 2026-09 i TextSight (bez treningu, deduplikacja).
- **Kontrakt:** kev-suites / typed-decisions — model nie może być wyraźnie gorszy od Kev-bazy.
- Metryki: AUC, **TPR przy FPR 1%**, fałszywe alarmy przy progu domyślnym, ECE i Brier, osobno na
  każdym wycinku; latencja i VRAM.

Kryteria akceptacji (vs v5), ustalone teraz, przed treningiem:

| miara | v5 | cel |
|---|---|---|
| fałszywe alarmy, eseje akademickie (dev, IvyPanda, `essays_2021`) | 0–0,7% | ≤ 0,5% |
| fałszywe alarmy, osoby uczące się angielskiego (CSEE / ICNALE) | 42% CSEE | **≤ 5%** |
| AI wykryte, rodziny wyłączone z treningu | 100% (za łatwe) | ≥ 95% na nowym, trudniejszym zestawie |
| AI wykryte, eseje po humanizacji / AI-edycji | brak pomiaru | ≥ 70% |
| WasItAI: ludzkie → AI / surowe AI wykryte | 16,8% / 61% | ≤ 5% / ≥ 60% |
| ECE (kalibracja) na walidacji | brak pomiaru | ≤ 0,05 |

### Faza 5 — wdrożenie i publikacja

- Aplikacja wywołuje lokalny serwer Kev (`/v1/systemone`) lub bezpośrednio model w Pythonie;
  zwraca nie tylko p_ai, ale też `origin` i `ai_share`. v5 jako fallback (np. gdy brak GPU).
- Inferencja: 4B w 4-bit ok. 3,5 GB VRAM; na samym CPU 4B jest wolny (kilka–kilkanaście s na esej),
  więc dla komputerów bez GPU wariant 0.8B.
- Publikacja adaptera LoRA na HF: **licencja wag wynika z licencji danych** (PERSUADE, ELLIPSE są
  niekomercyjne), więc najpewniej CC BY-NC; do decyzji po fazie 1.
- Dokumentacja: WYNIKI.md, LaTeX, karty danych, karta modelu z opisem ograniczeń.

## 4. Harmonogram (orientacyjny)

| tydzień | praca | GPU w nocy |
|---|---|---|
| 1 | faza 0, start fazy 1 (pobieranie, karty danych) | generowanie esejów |
| 2 | faza 1 (czyszczenie, przecieki), faza 2 | generowanie + Kev-0.8B pilot |
| 3 | faza 3: iteracje 0.8B, hard mining | trening 0.8B |
| 4 | faza 3: Kev-4B QLoRA, faza 4 | trening 4B (2–3 noce) |
| 5 | faza 4 (finalny test raz), faza 5, dokumentacja | — |

## 5. Ryzyka

| ryzyko | co robimy |
|---|---|
| Gated DeltaNet / Triton / bitsandbytes nie działa na Windows | WSL2 od początku; test w fazie 0 |
| 4B nie mieści się lub jest za wolny | model finalny 0.8B, więcej danych zamiast większego modelu |
| model uczy się źródła (formatowanie, długość, temat) zamiast stylu | normalizacja, dopasowanie długości, te same tematy human/AI, testy na rodzinach wyłączonych |
| znów stronniczość wobec osób nieanglojęzycznych | ICNALE/ELLIPSE w treningu, osobne cele akceptacji, wycinki w raporcie |
| licencje (NC, ICNALE na wniosek) | karta danych przed użyciem; bez licencji — tylko ewaluacja albo pomijamy |
| douczenie psuje ogólny kontrakt Kev | replay + kev-suites jako bramka |
| przeciek przez wielokrotne użycie testów | nowy zbiór finalny, uruchamiany raz; `essays_2021` tylko regresja |
| fakty z README (rozmiary, VRAM, licencje) okażą się nieaktualne | wszystko weryfikowane w fazie 0 |

## 6. Pierwsze konkretne kroki

1. Zainstalować WSL2 + Ubuntu 24.04 + sterownik CUDA dla WSL, stworzyć osobne `.venv-s1`.
2. Pobrać `jaredpalmer/kev-0.8b`, uruchomić serwer Kev, wysłać testowe żądanie Noul z `jev_client.py`.
3. Skrypt `train/s1_spike.py`: pomiar VRAM / tok/s (0.8B bf16, 4B NF4; seq 1k/2k/4k; inferencja i krok LoRA).
4. Zero-shot Kev na `data/dev` i `data/prepared_v5_tests` → tabela obok v5.
5. Zamrozić protokół z fazy 4 (lista zbiorów testowych + kryteria) w commicie, **zanim** ruszy trening.

## Źródła

- TypeSafe, Noul: https://docs.typesafe.ai/primitives/noul
- Kev: https://github.com/jaredpalmer/kev
- Nimble: https://github.com/bespokelabsai/nimble
- Laya: https://github.com/NandhaKishorM/laya
- Von (archiwum): https://github.com/wfzyx/von
- system-one-open: https://github.com/mithalouni/system-one-open
- open-alternative-jev: https://github.com/ikermoel/open-alternative-jev
- przegląd alternatyw: https://dev.to/yanng981/open-source-jev-alternatives-system-one-models-you-can-self-host-i5e ,
  https://www.datacamp.com/blog/top-open-source-jev-alternatives
- kontrakt API (adapter): https://github.com/rcarmo/go-system-one/blob/main/docs/systemone-api.md
