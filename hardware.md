# Hardware — RAG Chatbot & Agente AI Interno

*Stack attuale: FastAPI + BGE-M3 + BGE-reranker-v2-m3 + Ollama/vLLM + Qdrant + LlamaIndex AgentWorkflow*
*Budget: ~€10.000 | Data: aprile 2026*

---

## Strategia: Server Unico, Modello Unico

Concentrare l'intero budget su una sola macchina ed eseguire un unico modello 70B sia per il chatbot che per lo strumento agente interno è la scelta migliore. I motivi:

- **Un solo caricamento del modello** — nessuna VRAM sprecata caricando due modelli separati su due macchine
- **Pool di inferenza condiviso** — entrambi i carichi di lavoro si accodano allo stesso motore vLLM; la capacità inattiva del chatbot è disponibile per l'agente e viceversa
- **Qualità del modello superiore** — l'intero budget va nelle GPU, permettendo un modello 70B invece di 32B, un salto significativo per il ragionamento agente
- **Operatività semplificata** — un solo server da gestire, un'istanza Qdrant, un processo vLLM

Il chatbot e lo strumento agente rimangono app FastAPI separate (o endpoint separati sulla stessa app). Chiamano la stessa API LLM con system prompt e configurazioni skill diverse — esattamente come il routing `skills/` già funziona nel codice attuale.

---

## Configurazione Finale

| Componente | Prodotto |
|---|---|
| **CPU** | AMD Ryzen Threadripper PRO 9955WX (16c/32t, 4.5–5.4 GHz, 64 MB cache) |
| **Scheda madre** | ASUS PRO WS WRX90E-SAGE SE (WRX90, ECC RDIMM, PCIe 5.0) |
| **RAM** | 128 GB DDR5 Kingston 6400 MHz ECC RDIMM (2× 64 GB) |
| **GPU** | 2× NVIDIA RTX 5090 32 GB GDDR7 Founders Edition |
| **Storage** | Samsung 9100 Pro M.2 PCIe 5.0 NVMe 2 TB (14.700 MB/s lettura) |
| **Alimentatore** | FSP Cannon Pro 2500W ATX 3.1 80 PLUS Platinum modulare |
| **Case** | FSP U530 Tower (verificare compatibilità dual triple-slot — vedi nota) |
| **Raffreddamento CPU** | Silverstone XE360-TR5 AIO 360mm (socket TR5 nativo) |
| **Raffreddamento RAM** | Dissipatore integrato ASUS WRX90 |
| **TPM** | Asus TPM-SPI |
| **Sistema operativo** | Ubuntu Server 24.04 LTS |
| **Garanzia** | Oro 3 anni (2 anni ritiro/spedizione, 2 anni ricambi, 3 anni manodopera) |

> **Nota case**: La RTX 5090 Founders Edition è una scheda tripla-slot da ~336 mm. Prima di confermare l'ordine, verificare con il builder che il FSP U530 supporti due schede triple-slot installate simultaneamente con ventilazione adeguata. Alternative sicure: **Fractal Define 7 XL** o **Lian Li O11 Dynamic EVO XL**.

---

## Perché Threadripper PRO invece di Ryzen 9

Il nodo critico con due RTX 5090 è la banda PCIe durante l'inferenza tensor-parallel. Su piattaforma consumer (Ryzen 9 + X870E), il secondo slot PCIe fisicamente sembra x16 ma gira **elettricamente a x8 o x4** — il Ryzen 9 consumer ha solo 24 lane PCIe totali da condividere tra GPU, NVMe e chipset.

Il **Threadripper PRO 9955WX ha 128 lane PCIe 5.0 native dalla CPU**. Entrambe le 5090 girano a PCIe 5.0 x16 reale. Nell'inferenza tensor-parallel le due GPU si scambiano continuamente gli attivatori dei layer — una banda ridotta crea un collo di bottiglia che nessuna quantità di VRAM risolve.

### Perché RAM ECC RDIMM

La piattaforma WRX90 con Threadripper PRO **supporta solo RDIMM ECC** — le UDIMM (ECC non-registered o standard) non sono compatibili con questa scheda madre. Le RDIMM aggiungono un chip "registro" sul modulo che fa da buffer tra i pin e il controller, permettendo configurazioni più grandi e stabili, obbligatorie sulle piattaforme workstation. La latenza leggermente superiore rispetto alle UDIMM è irrilevante per l'inferenza LLM.

Se in futuro si vuole espandere a 256 GB, basta aggiungere altri 2 moduli da 64 GB dello stesso kit — la WRX90 ha 8 slot RDIMM.

---

## Ollama vs vLLM

Entrambi caricano il modello in GPU e rispondono alle richieste, ma con architetture interne completamente diverse.

### Ollama (setup iniziale consigliato)

Gestisce le richieste una alla volta in sequenza. Mentre la GPU lavora per l'utente 1, l'utente 2 aspetta in coda. Configurazione immediata, zero complessità operativa.

### vLLM (upgrade quando il carico cresce)

Introduce due tecnologie chiave:

- **Continuous batching**: raggruppa i token di più richieste nello stesso ciclo GPU. Le nuove richieste entrano nel batch nel momento esatto in cui un'altra finisce — la GPU non aspetta mai tra richieste.
- **PagedAttention**: gestisce la KV-cache come pagine di RAM allocate dinamicamente. Ollama alloca un blocco fisso di VRAM per ogni richiesta anche se non viene usato tutto. vLLM alloca solo quello che serve, permettendo molte più conversazioni simultanee.

| | Ollama | vLLM |
|---|---|---|
| Richieste simultanee | 1 alla volta (coda) | 15–30 in parallelo reale |
| Utilizzo GPU | ~40–60% | ~85–95% |
| Throughput totale | base | 3–5× più alto |
| Latenza singola richiesta | bassa | leggermente più alta sotto carico leggero |
| Configurazione | `ollama serve` e basta | file di config, ~10 minuti |
| Compatibilità con `main.py` | attuale | **nessuna modifica** — stessa API OpenAI |

Il passaggio da Ollama a vLLM non richiede modifiche a `main.py` — cambia solo l'URL e il client LLM:

```python
# Attuale (Ollama)
llm = Ollama(model="qwen2.5:72b", request_timeout=120.0)

# Con vLLM
from llama_index.llms.openai_like import OpenAILike
llm = OpenAILike(model="qwen2.5:72b", api_base="http://localhost:8000/v1", api_key="none")
```

---

## Stima Utenti Simultanei

Modello: **Qwen2.5-72B-Instruct Q4_K_M** su 2× RTX 5090 (tensor parallel)

Velocità stimata token single-stream: ~70–90 tok/s (banda GDDR7 ~3,6 TB/s aggregati)

### Carico Chatbot

| Metrica | Valore |
|---|---|
| Lunghezza media risposta | ~300 token |
| Overhead retrieval + rerank | ~1,5–2 s |
| Tempo totale per query | ~5–7 s |
| **Utenti simultanei comodi (Ollama)** | **8–12** |
| **Con vLLM continuous batching** | **20–30** |

### Carico Agente

| Metrica | Valore |
|---|---|
| Lunghezza media task | ~1.000–1.500 token (4–8 chiamate LLM) |
| Overhead esecuzione strumenti | ~3–8 s per task |
| Tempo totale per task agente | ~25–50 s |
| **Utenti simultanei comodi (Ollama)** | **5–8** |
| **Con vLLM continuous batching** | **10–15** |

### Carico Misto

vLLM gestisce entrambe le code sullo stesso processo modello. Le richieste brevi del chatbot riempiono automaticamente gli spazi tra i task agente lunghi grazie al continuous batching.

| Mix di carico | Utenti simultanei stimati |
|---|---|
| Solo chatbot | 20–30 |
| Solo agente | 10–15 |
| Misto (70% chatbot / 30% agente) | **15–22 totali** |

> Stime con tempo di attesa accettabile ≤30 s. Il throughput reale dipende dalla lunghezza effettiva dei prompt e dall'utilizzo della finestra di contesto.

---

## Modello Consigliato

### Modello unico per entrambi i carichi di lavoro

**`Qwen2.5-72B-Instruct Q4_K_M`** — scelta principale.

| Proprietà | Valore |
|---|---|
| VRAM a Q4_K_M | ~42 GB (entra nei 64 GB con ~22 GB liberi per KV-cache) |
| Finestra di contesto | 128k token |
| Qualità lingua italiana | Eccellente (addestrato su dati multilingua) |
| Tool / function calling | Supporto nativo — compatibile con LlamaIndex AgentWorkflow |
| Seguire le istruzioni | Migliore della categoria per pesi aperti a scala 70B |

La stessa istanza del modello serve sia le route skill del chatbot che il workflow agente. Il routing `skills/` nel codice attuale gestisce già questo — system prompt e set di strumenti diversi, stesso LLM sottostante.

### Alternative

| Modello | VRAM (Q4) | Note |
|---|---|---|
| **Llama 3.3 70B Instruct** Q4_K_M | ~40 GB | Ragionamento solido, italiano leggermente inferiore a Qwen |
| **DeepSeek-R1 70B** Q4_K_M | ~42 GB | Chain-of-thought integrato; ottimo per task agente, più lento per query chatbot semplici |
| **Qwen2.5-32B-Instruct** Q4_K_M | ~20 GB | Fallback se il 70B risulta troppo lento; libera ~42 GB VRAM per KV-cache più ampia o un secondo modello futuro |

### Embedding & Reranker (nessun cambio consigliato)

I modelli attuali (BGE-M3 + BGE-reranker-v2-m3) sono già allo stato dell'arte per il retrieval ibrido e l'uso multilingua. Tenerli invariati.

---

## Comandi di Setup

```bash
# Scarica il modello
ollama pull qwen2.5:72b

# Abilita tensor-parallel su entrambe le 5090
CUDA_VISIBLE_DEVICES=0,1 ollama serve
```

Per una concorrenza maggiore, passare a vLLM:

```bash
pip install vllm
vllm serve Qwen/Qwen2.5-72B-Instruct \
  --tensor-parallel-size 2 \
  --quantization awq \
  --max-model-len 32768 \
  --gpu-memory-utilization 0.90
```

---

## Note Operative

- **Consumo elettrico**: 2× RTX 5090 a pieno carico = ~1.150 W GPU + ~150 W sistema. L'alimentatore da 2500W garantisce ampio margine. Si raccomanda un UPS minimo da 2000 VA per spegnimenti puliti.
- **Sistema operativo**: Ubuntu Server 24.04 LTS — miglior supporto driver NVIDIA, pacchetti Qdrant/Ollama nativi.
- **Rete**: Esporre il chatbot tramite reverse proxy nginx con rate limiting. Tenere lo strumento agente solo sulla LAN interna.

---

## Piano di Scaling Futuro

### Il problema da risolvere prima di scalare

La memoria di conversazione (`ChatMemoryBuffer`) è attualmente salvata **in memoria nel processo FastAPI**, in un dizionario indicizzato per `session_id`. Questo funziona su un singolo server, ma blocca qualsiasi scaling orizzontale: se una richiesta dello stesso utente arriva su un secondo server, la memoria della conversazione non c'è.

Prima di aggiungere un secondo server è necessario **esternalizzare la memoria su Redis**. È una modifica chirurgica a `main.py`: invece di un dizionario Python si usa un client Redis che serializza/deserializza la lista di messaggi per session_id. Il resto del codice rimane invariato.

---

### Architettura con due server

```
                        Internet / LAN
                              │
                    ┌─────────▼─────────┐
                    │       nginx        │
                    │  (load balancer)   │
                    │     least_conn     │
                    └────┬──────────┬────┘
                         │          │
               ┌─────────▼──┐  ┌───▼──────────┐
               │  FastAPI 1  │  │  FastAPI 2   │
               │  BGE-M3     │  │  BGE-M3      │
               │  Reranker   │  │  Reranker    │
               │  (Server 1) │  │  (Server 2)  │
               └──────┬──────┘  └──────┬───────┘
                      │                │
          ┌───────────┼────────────────┤
          │           │                │
    ┌─────▼─────┐  ┌──▼────────┐  ┌───▼──────┐
    │   Redis   │  │  Qdrant   │  │ LiteLLM  │
    │ (memoria  │  │ (vettori, │  │  Proxy   │
    │ sessioni) │  │  unica    │  │  (LLM LB)│
    └───────────┘  │  istanza) │  └──┬────┬──┘
                   └───────────┘     │    │
                                ┌────▼┐  ┌▼────┐
                                │vLLM │  │vLLM │
                                │ S1  │  │ S2  │
                                └─────┘  └─────┘
```

---

### Componenti dello stack di scaling

#### 1. nginx — bilanciamento HTTP esterno

nginx riceve tutte le richieste HTTP e le distribuisce tra i due server FastAPI.

La strategia `least_conn` è preferibile al round-robin semplice perché le richieste LLM hanno durata molto variabile (una query chatbot dura ~5 s, un task agente ~40 s): `least_conn` manda la prossima richiesta al server con meno connessioni aperte, evitando di sovraccaricare quello già impegnato su task lunghi.

```nginx
upstream rag_backend {
    least_conn;
    server server1:8000;
    server server2:8000;
    keepalive 32;
}

server {
    listen 80;
    location / {
        proxy_pass http://rag_backend;
        proxy_read_timeout 120s;
    }
}
```

#### 2. Redis — memoria di sessione condivisa

Entrambi i server FastAPI leggono e scrivono la cronologia conversazione su Redis invece che in un dizionario locale. La lista dei messaggi per `session_id` viene serializzata come JSON su una chiave Redis con TTL (es. 2 ore di inattività).

In questo modo una conversazione iniziata su Server 1 può continuare su Server 2 senza perdere il contesto.

```bash
docker run -d --name redis -p 6379:6379 redis:7-alpine
```

#### 3. LiteLLM Proxy — bilanciamento delle inferenze LLM

LiteLLM è un proxy che espone una singola API compatibile OpenAI e bilancia le richieste tra i due endpoint vLLM con strategia `least-busy` — ideale per LLM dove la durata delle richieste è imprevedibile.

Vantaggi rispetto a nginx direttamente davanti a vLLM:
- conosce il concetto di "slot liberi" nei modelli LLM, non solo le connessioni TCP
- gestisce retry automatici se un server è sovraccarico
- logging centralizzato dei token consumati

```yaml
# litellm_config.yaml
model_list:
  - model_name: qwen2.5-72b
    litellm_params:
      model: openai/qwen2.5:72b
      api_base: http://server1:8000/v1
      api_key: none
  - model_name: qwen2.5-72b
    litellm_params:
      model: openai/qwen2.5:72b
      api_base: http://server2:8000/v1
      api_key: none

router_settings:
  routing_strategy: least-busy
```

```bash
litellm --config litellm_config.yaml --port 4000
```

`main.py` punta su `http://litellm-host:4000/v1` — nessuna altra modifica necessaria.

#### 4. Qdrant — nessuna modifica necessaria

Qdrant può rimanere su un'unica istanza. Entrambi i server FastAPI la leggono in parallelo senza conflitti — le query di ricerca sono read-only al 99%. Supporta anche una modalità cluster distribuita nativa se in futuro il volume di documenti dovesse crescere enormemente.

---

### Stima utenti con due server

Con due server identici (2× RTX 5090 ciascuno) e vLLM + LiteLLM:

| Mix di carico | 1 server | 2 server |
|---|---|---|
| Solo chatbot | 20–30 | **40–55** |
| Solo agente | 10–15 | **20–28** |
| Misto (70/30) | 15–22 | **30–42 totali** |

Lo scaling è quasi lineare perché il collo di bottiglia rimane l'inferenza GPU — i componenti condivisi (Redis, Qdrant, nginx) gestiscono carichi molto più elevati senza diventare un problema.

---

### Ordine consigliato degli step

| Step | Quando farlo | Costo |
|---|---|---|
| 1. Passare da Ollama a vLLM | Subito (guadagno immediato di concorrenza, stesso hardware) | €0 |
| 2. Aggiungere Redis per le sessioni | Prima di acquistare il secondo server | €0 (software) |
| 3. Acquistare secondo server identico | Quando il carico supera stabilmente 15 utenti simultanei | ~€10.000 |
| 4. Installare LiteLLM Proxy + riconfigurare nginx | Contestualmente al passo 3 | €0 (software) |
