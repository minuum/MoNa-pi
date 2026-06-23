# Ablation Pipeline 결과 (M6 → M5 → M7)

> 실행: 2026-06-23 18:00~20:47 (minum 서버, `monapi-train` 브랜치)
> `scripts/run_ablation.sh` (2026-06-02 작성, 이번에 처음 실행) — 중간에 브랜치 전환 실수로
> M5 eval 단계가 한 번 끊겼다가(`scripts/eval_closedloop.py`가 `monapi-driving`에는 없는 파일이라
> working tree에서 사라짐) `monapi-train`에 고정한 채 재실행해 완료.

## M6 — ODE steps 3/5/10 (재학습 없음, eval만)

베이스라인(`checkpoints/best`)으로 ODE 적분 스텝만 바꿔 품질/속도 트레이드오프 확인.

| ODE steps | SR (regular, n=38) | FPE |
|---|---|---|
| 3 | 100.0% | 0.0707m |
| 5 (현재 기본값) | 100.0% | 0.0697m |
| 10 | 100.0% | 0.0703m |

**결론**: 3/5/10 전부 거의 동일 — **ODE steps를 3으로 줄여도 품질 손해 없음**. 추론 속도를 줄일 수 있는 여유가 있다는 뜻 (현재 기본 5 대비 추가 단축 가능, 실측 latency 비교는 별도 필요).

## M5 — 증강 없이 학습 (244ep, mobile_vla_dataset_merged) vs baseline(증강 있음)

| 구성 | eval 대상 | SR | FPE |
|---|---|---|---|
| baseline (증강 있음, `checkpoints/best`) | full (n=53) | 73.6% | 0.3906m |
| M5 (증강 없음) | full (n=53) | 73.6% | 0.3868m |
| M5 (증강 없음) | regular (n=38) | 100.0% | 0.0694m |

baseline의 `full` eval도 추가로 돌려서 직접 비교함 — **증강 유/무가 SR·FPE에 거의 차이를 만들지 않음(73.6%/73.6%, FPE 차이 0.004m로 노이즈 수준)**.

### 14건 실패의 정체 — `episode_idx`를 실제 파일명으로 역추적 (정정)

처음엔 "`full`에만 포함되는 나머지 15개 에피소드가 center_left/center_right/straight 등 약한 카테고리일 것"으로 추정했으나, `episode_idx`를 실제 h5 파일명으로 역매핑해 확인한 결과 **완전히 다른 그림**이었다.

| 카테고리 | 구성 | 실패 |
|---|---|---|
| 정형 9종(center/left/right × straight/left/right) | 38개 | **0개 (0%)** |
| `free_center_*` (basket 극단오프셋/대각선/로봇거리/조명) | 5개 | 4개 (1개만 성공: robot_far) |
| `free_left_*` | 5개 | **5개 (100%)** |
| `free_right_*` | 5개 | **5개 (100%)** |

baseline과 M5(no-aug) 둘 다 동일하게 **free fail/total = 14/15, regular fail = 0/38**.

✅ **정정된 결론**: 약점은 "center/straight 카테고리"가 아니라 **`free_*` 의도적 OOD 프로브 에피소드 전체**(basket 극단 오프셋, 대각선 시작, 로봇 원거리, 조명 변화 — 거의 모든 변형에서 실패). 정형 9종 경로는 (증강 유무와 무관하게) 100% 성공하므로, CH7의 add_free 244ep 통합(center_straight 21ep 추가 등)이 실제로 "정형 경로" 문제는 해결한 것으로 보인다. 남은 문제는 augmentation으로 풀리는 종류가 아니라 **학습 분포 밖(OOD) 일반화 자체의 한계** — `free_*` 부류 자체를 학습 데이터에 더 포함시키거나, OOD에 강한 정규화/증강(예: 극단 오프셋·대각선 시작을 흉내내는 합성 증강)이 필요할 것으로 보임.

## M7 — 116ep 원본 v5 데이터만 (mobile_vla_dataset_v5, 증강 없음)

| eval 대상 | SR | FPE |
|---|---|---|
| regular (n=24) | 100.0% | 0.0485m |
| full (n=24) | 100.0% | 0.0485m |

⚠️ **해석 주의**: M7의 `regular`와 `full`이 완전히 동일(둘 다 n=24)하다는 건, **116ep 원본 v5 데이터셋 자체에 "hard" 카테고리 평가 에피소드가 거의/전혀 없다는 뜻** — 즉 "원본 데이터만으로도 100%"가 "작은 데이터셋이 더 낫다"를 의미하지 않음. 단지 그 데이터셋의 eval 셋이 쉬운 케이스로만 구성돼 있어서 어려운 케이스를 아예 평가하지 않은 것. M5(244ep, full=53)와 M7(116ep, full=24)을 직접 비교하는 건 **apples-to-apples가 아님** — 서로 다른 eval 모집단.

## 종합

- ✅ **ODE steps 축소(3)는 안전** — 바로 적용 가능한 최적화.
- ✅ **약점은 `free_*` OOD 프로브 전체(정형 9종 경로는 100%)** — baseline과 M5(no-aug)가 73.6%/73.6%로 동일하고, 실패 14건 전부 `free_*`(14/15), 정형 경로 실패 0건(0/38)으로 정확히 일치. 증강으로는 안 풀리는 OOD 일반화 문제.
- ❓ **"116ep로도 충분하다"는 결론은 못 냄** — M7의 eval 모집단(n=24)이 애초에 쉬운 케이스로만 구성돼 있어 M5(n=53)와 직접 비교 불가(apples-to-apples 아님). 116ep 데이터셋에도 244ep 수준의 hard-category 평가셋을 적용해야 진짜 비교 가능 — 후속 작업으로 남김.
