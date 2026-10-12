# Do vídeo à trajetória: detecção, tracking e calibração em transmissões de futebol

#### Aluno: [Matheus de Amorim](https://github.com/mthsdamorim)
#### Orientadora: [Manoela](mailto:prof.manoela@ica.ele.puc-rio.br)

---

Trabalho apresentado ao curso [VC MASTER](https://ica.puc-rio.ai/vc-master) como pré-requisito para conclusão de curso e obtenção de crédito na disciplina "Projetos de Sistemas Inteligentes de Apoio à Decisão".

- [Link para o código](packages/).

- [Link para a monografia](monografia.pdf).

- Trabalhos relacionados:
    - [SoccerNet Game State Reconstruction: End-to-End Athlete Tracking and Identification on a Minimap](https://arxiv.org/abs/2404.11335).
    - [PnLCalib: Sports Field Registration via Points and Lines Optimization](https://arxiv.org/abs/2404.08401).
    - [ByteTrack: Multi-Object Tracking by Associating Every Detection Box](https://arxiv.org/abs/2110.06864).
    - [RF-DETR: Neural Architecture Search for Real-Time Detection Transformers](https://arxiv.org/abs/2511.09554).

---

### Resumo

Este trabalho investiga até onde é possível obter observações posicionais úteis a partir de uma transmissão de futebol com uma única câmera. O protótipo detecta pessoas com RF-DETR-L, associa as detecções entre quadros com ByteTrack, estima a câmera com PnLCalib e projeta o ponto de apoio de cada pessoa no plano do campo, abstendo-se de produzir coordenadas métricas quando não há solução de câmera. Nenhum modelo foi treinado: o trabalho consistiu em integrar componentes pré-treinados, preservar o tempo do vídeo e estudar as falhas por meio de experimentos com critérios definidos antes da execução. A avaliação final usou três janelas reservadas de dez segundos do SoccerNet Game State Reconstruction (GSR). Nos 50 quadros amostrados por clipe, a cobertura de pessoas variou de 0,805 a 0,938 e o IDF1, de 0,792 a 0,899. A câmera foi estimada em 250, 250 e 208 dos 250 quadros de cada clipe, e o último caso concentrou os maiores desacordos geométricos na cauda da distribuição. A suavização temporal não melhorou consistentemente os resultados de cauda. Os valores em metros descrevem concordância com o subconjunto de apoios revisados do GSR, sem demonstrar precisão física absoluta.

### Abstract

This work investigates how far useful positional observations can be obtained from a single-camera football broadcast. The prototype detects people with RF-DETR-L, associates detections across frames with ByteTrack, estimates the camera with PnLCalib and projects each person's ground-support point onto the pitch plane, abstaining from metric coordinates whenever no camera solution exists. No model was trained: the work consisted of integrating pretrained components, preserving video time and studying failures through experiments whose criteria were fixed before execution. The final evaluation used three held-out ten-second windows from SoccerNet Game State Reconstruction (GSR). Over the 50 sampled frames per clip, person coverage ranged from 0.805 to 0.938 and IDF1 from 0.792 to 0.899. The camera was estimated in 250, 250 and 208 of the 250 frames of each clip, and the last clip concentrated the largest geometric disagreements in the tail of the distribution. Temporal smoothing did not consistently improve tail results. Metric values describe agreement with the reviewed GSR support subset and do not demonstrate absolute physical accuracy.

### 1. Introdução

O movimento observado num vídeo de transmissão combina o deslocamento das pessoas no campo com o giro e o zoom da câmera. A transmissão mostra só parte do jogo: pessoas saem do enquadramento, se sobrepõem e ocupam poucos pixels. Por isso, a distância entre duas caixas na imagem não equivale ao deslocamento no campo.

O objetivo foi construir e avaliar um protótipo, sem exigência de processamento em tempo real, que detecte e associe as pessoas visíveis em trechos curtos e as projete no campo quando houver câmera aceita. Jogadores de linha, goleiros e árbitros são tratados como pessoas observadas; equipe, papel e identidade nominal ficam fora da avaliação, assim como a bola, a continuidade entre cortes e a reconstrução de pessoas fora do quadro.

### 2. Modelagem

O método separa as etapas para permitir a inspeção de cada uma:

1. **Vídeo:** decodificação dos quadros preservando o instante de apresentação (PTS) e a base de tempo. No GSR, que fornece JPEGs, o tempo é derivado como `índice / 25`.
2. **Pessoas:** RF-DETR-L pré-treinado produz caixas (confiança mínima 0,1); ByteTrack associa as observações em IDs válidos apenas dentro da sessão. O centro inferior da caixa aproxima o ponto de apoio no solo.
3. **Câmera:** PnLCalib estima, a cada quadro, uma homografia a partir de pontos e linhas do campo, ou registra abstenção.
4. **Projeção:** apoio e homografia produzem uma coordenada num campo assumido de 105 × 68 m. Sem câmera aceita, a caixa e o ID permanecem, mas a coordenada métrica fica ausente.
5. **Avaliação:** os avaliadores leem os rótulos só depois da inferência. A identidade é medida com pareamento por IoU ≥ 0,5 (cobertura, IDF1, trocas e fragmentações); a geometria, com disponibilidade da câmera, erro de linhas em pixels e diferença em metros nos apoios revisados. Uma variante suavizada (Savitzky–Golay, janela 5, grau 2) é comparada sem substituir a observação original.

Antes da avaliação final, o desenvolvimento passou por experimentos de propagação temporal da câmera, refinamento espacial por linhas, transferência para novos clipes, revisão manual dos pontos de apoio, suavização e atribuição de equipe e papel. Vários deles não atingiram os critérios definidos previamente e foram mantidos como resultados negativos.

A amostra final usa três clipes de jogos distintos da partição de teste do SoccerNet-GSR (`SNGS-116`, `SNGS-132` e `SNGS-187`), quadros 251 a 500 de cada um. A identidade é avaliada em 50 quadros por clipe, amostrados a 5 fps, e 10 desses quadros por clipe passaram por revisão humana.

#### Organização do código

```text
packages/
  football_alpha_video/              decodificação RGB preservando PTS e time_base
  football_alpha_player_detection/   detecção de pessoas (RF-DETR-L)
  football_alpha_person_tracking/    associação local (ByteTrack) e diagnóstico
  football_alpha_pitch_projection/   geometria do campo, calibração e projeção
scripts/
  prepare-tcc-reference.py           monta e valida a referência GSR reservada
  render-tcc-review.py               gera as imagens da revisão humana
  pnlcalib-experiment.py             estima a câmera (PnLCalib) e projeta os tracks
  evaluate-tcc-identity.py           cobertura, IDF1, trocas e fragmentações
  evaluate-tcc-projection.py         câmera, erro de linhas, metros e suavização
  tcc_protocol_matching.py           pareamento de caixas compartilhado
evaluation/                          protocolo congelado, revisão humana e resultados
tests/                               testes (cobertura de 100%)
```

### 3. Resultados

| Clipe | Cobertura | IDF1 | Trocas de ID | Câmera / 250 | Linhas: mediana / p90 (px) | Diferença: mediana / p90 (m) |
| --- | --- | --- | --- | --- | --- | --- |
| SNGS-116 | 0,938 | 0,899 | 10 | 250 | 2,64 / 6,25 | 0,82 / 2,28 |
| SNGS-132 | 0,805 | 0,792 | 8 | 250 | 3,81 / 11,34 | 0,89 / 3,62 |
| SNGS-187 | 0,919 | 0,814 | 14 | 208 | 3,27 / 140,04 | 0,58 / 19,20 |

SNGS-116 teve o melhor resultado de identidade local. Em SNGS-132, 148 das 758 pessoas de referência ficaram sem par. SNGS-187 concentrou os maiores desacordos geométricos: com mediana de 3,27 px no erro de linhas, o p90 chega a 140,04 px, o que mostra que uma câmera aceita pelo método ainda pode errar. Nesse clipe, 42 quadros ficaram sem câmera e 745 observações permaneceram apenas em pixels, sem projeção forçada. A suavização reduziu a oscilação mediana, mas piorou o p90 de oscilação em SNGS-132 e SNGS-187 e o p90 da diferença em SNGS-132 (de 3,62 para 3,82 m).

Os resultados completos estão em [`evaluation/tcc-identity-result.json`](evaluation/tcc-identity-result.json) e [`evaluation/tcc-projection-result.json`](evaluation/tcc-projection-result.json).

### 4. Conclusões

O protótipo integra detecção, associação temporal e projeção no campo, com diagnósticos que permitem inspecionar perdas, trocas de ID e falhas de câmera. Cobertura e IDF1 mediram a associação de pessoas nas três janelas reservadas; as abstenções e os desacordos geométricos limitaram o uso das posições projetadas. A disponibilidade de uma solução de câmera não garante sua qualidade, e a suavização não trouxe benefício uniforme.

As contribuições são: um protótipo que preserva observações em imagem, tempo e IDs locais; uma política de projeção com abstenção quando falta câmera aceita; um histórico de experimentos com critérios definidos, controles e resultados negativos preservados; e um pacote de avaliação descritiva com fontes, revisão humana e denominadores explícitos.

Três trechos de dez segundos não sustentam extrapolação para partidas completas, outras ligas ou outras câmeras, e a referência do GSR não é uma aferição física independente. Antes de medir distâncias percorridas ou carga física, seria necessário qualificar geometria, ponto de apoio e identidade com referência independente.

### 5. Reprodução

Requer [uv](https://docs.astral.sh/uv/), Python 3.13 e FFmpeg.

```bash
uv sync --all-packages --all-groups --locked
scripts/check.sh   # ruff, pyright e pytest
```

**Dados.** Baixe `test.zip` do [SN-GSR-2024](https://huggingface.co/datasets/SoccerNet/SN-GSR-2024) na revisão `3cc710eb6d53a23350a3d2863311c1c0a2e645d7` e copie, para cada clipe, o `Labels-GameState.json` e os JPEGs 251 a 500 para `artifacts/issue18/candidate-review/<clip>/img1/`. Depois:

```bash
mkdir -p artifacts/issue18/derived
for clip in SNGS-116 SNGS-132 SNGS-187; do
  jq --arg clip "$clip" '.clips[$clip]' evaluation/tcc-reference-sources.json \
    > "artifacts/issue18/candidate-review/$clip/candidate-manifest.json"
  ffmpeg -hide_banner -loglevel error -n -framerate 25 -start_number 251 \
    -i "artifacts/issue18/candidate-review/$clip/img1/%06d.jpg" \
    -frames:v 250 -c:v libx264rgb -crf 0 -preset veryfast -pix_fmt rgb24 \
    -video_track_timescale 25000 "artifacts/issue18/derived/$clip.mp4"
done
uv run python scripts/prepare-tcc-reference.py \
  --protocol evaluation/tcc-reference-protocol.json \
  --source-root artifacts/issue18/candidate-review \
  --video-root artifacts/issue18/derived \
  --output artifacts/issue18/reference
```

**Pesos do PnLCalib.** O RF-DETR-L é baixado automaticamente pelo pacote `rfdetr`.

```bash
mkdir -p artifacts/pnlcalib/vendor artifacts/pnlcalib/weights
git clone https://github.com/mguti97/PnLCalib.git artifacts/pnlcalib/vendor/PnLCalib
git -C artifacts/pnlcalib/vendor/PnLCalib checkout --detach 8c87391d6f4ea40c5e4d65e61529916c7a49ce62
curl -fL https://github.com/mguti97/PnLCalib/releases/download/v1.0.0/SV_kp -o artifacts/pnlcalib/weights/SV_kp
curl -fL https://github.com/mguti97/PnLCalib/releases/download/v1.0.0/SV_lines -o artifacts/pnlcalib/weights/SV_lines
```

**Inferência e avaliação.**

```bash
for clip in SNGS-116 SNGS-132 SNGS-187; do
  uv run football-alpha-person-tracking diagnose "artifacts/issue18/derived/$clip.mp4" \
    --composite "artifacts/issue19/$clip/tracking.mp4" \
    --jsonl "artifacts/issue19/$clip/tracks.jsonl" \
    --summary "artifacts/issue19/$clip/tracking-summary.json" \
    --batch-size 4
  uv run --with shapely==2.1.2 python scripts/pnlcalib-experiment.py \
    --source "artifacts/issue18/derived/$clip.mp4" \
    --tracks "artifacts/issue19/$clip/tracks.jsonl" \
    --tracking-summary "artifacts/issue19/$clip/tracking-summary.json" \
    --output "artifacts/issue19/$clip/camera"
done
uv run python scripts/evaluate-tcc-identity.py \
  --tracking-root artifacts/issue19 --output artifacts/eval/identity
uv run python scripts/evaluate-tcc-projection.py --output artifacts/eval/projection
cmp artifacts/eval/identity/summary.json evaluation/tcc-identity-result.json
cmp artifacts/eval/projection/summary.json evaluation/tcc-projection-result.json
```

Os avaliadores recusam entradas cujos hashes divergem do protocolo congelado. Vídeos, imagens do GSR e pesos não são versionados; respeite os termos de uso de cada fonte. Os vídeos de desenvolvimento estão descritos em [`data/README.md`](data/README.md). Código sob licença [MIT](LICENSE).

---

Matrícula: 242.100.205

Pontifícia Universidade Católica do Rio de Janeiro

Curso de Pós Graduação *Visão Computacional Master*
