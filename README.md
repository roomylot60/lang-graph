# LangGraph

### 1) 핵심 구조
- State(상태): 그래프 실행 중 공유되는 데이터 컨테이너(TypedDict 권장)
- Node(노드): state를 받아 일부 키만 갱신해 반환하는 함수
- Edge(간선): 실행 순서 연결 (선형/분기/루프)
- Conditional Edges: 조건 함수의 결과 라벨에 따라 다음 노드 선택
- Checkpointer: 중간 상태 스냅샷 저장/복구
- Reducer: messages처럼 누적/병합 규칙이 필요한 키에 부여하는 결합 함수

### 2) State 타입과 Reducer 설계
- LangGraph는 부분 갱신(Partial Update) 원칙
- 각 노드는 바뀐 키만 반환하면 되고, 나머지는 이전 상태가 유지
- 대화형에서는 messages 필드에 리듀서를 지정
```python
from typing import TypedDict, Annotated, Dict, List, Optional
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import AnyMessage, add_messages

# 상태 스키마 정의 (권장: TypedDict)
class State(TypedDict):
    # 대화 메시지: add_messages 리듀서로 자동 병합
    messages: Annotated[List[AnyMessage], add_messages]

    # STT→문장 버퍼(슬라이딩 윈도우)
    buffer_sentences: List[str]

    # 누적 요약(항상 고정 규격 유지: 예, 불릿 5개)
    running_summary: str

    # 경계 중복 방지용 마지막 1~2문장
    last_overlap: List[str]

    # 진단/메타 로깅
    meta: Dict[str, Optional[float]]
```

### 3) 노드(Node) 만들기
- 각 노드는 `state: State`를 받아 변경할 키만 반환
```python
# 1) STT 문장 수신 → 버퍼에 추가
def ingest_sentence(state: State) -> dict:
    # 예: meta에 수신 카운터 기록
    cnt = (state.get("meta", {}).get("recv_cnt") or 0) + 1
    new_meta = dict(state.get("meta", {}), recv_cnt=cnt)
    return {"buffer_sentences": state["buffer_sentences"], "meta": new_meta}
    # 실제 구현에선 외부 입력(신규 문장)을 인자로 받거나, 호출부에서 state에 미리 넣습니다.

# 2) 중복/경계 정리
def dedup_gate(state: State) -> dict:
    overlap = state.get("last_overlap", [])[-2:]  # 최근 2문장
    buf = state["buffer_sentences"]
    deduped = [s for s in buf if s not in overlap and len(s.strip()) > 0]
    return {"buffer_sentences": deduped}

# 3) 요약 트리거 판단 (조건 분기에서 사용)
def should_summarize(state: State) -> str:
    return "do" if len(state["buffer_sentences"]) >= 6 else "skip"

# 4) 컨텍스트 압축(누적 요약 → 1~2문장)
def compress_context(state: State) -> dict:
    prev = state.get("running_summary", "")
    # 실제로는 LLM/규칙으로 2문장 압축. 여기선 자리표시.
    context_2sents = prev[:200]  # 예시
    return {"meta": dict(state.get("meta", {}), ctx_len=len(context_2sents)),
            "running_summary": prev,  # 유지
            "buffer_sentences": state["buffer_sentences"],
            "last_overlap": state.get("last_overlap", [])}

# 5) 롤링 요약(고정 형식/길이)
def summarize_rolling(state: State) -> dict:
    # 입력: state["running_summary"]의 '의미만' 반영한 1~2문장(바로 전 노드 산출물로 전달하는 편)
    # + 최신 6문장으로 요약 생성 → 항상 불릿 5개, ≤80토큰 규격
    new_summary = "• 한줄요약...\n• 고객요청...\n• 상담사조치...\n• 민감슬롯...\n• 다음단계..."
    # 실제 구현에서는 모델 호출/토큰 제한을 적용
    return {"running_summary": new_summary}

# 6) 상태 갱신(버퍼 flush + overlap 업데이트)
def update_state(state: State) -> dict:
    buf = state["buffer_sentences"]
    new_overlap = buf[-2:] if len(buf) >= 2 else buf
    return {"buffer_sentences": [], "last_overlap": new_overlap}
```

### 4) 그래프(Edges) 구성
- 선형 연결 + 조건 분기를 섞어 실행 흐름 작성
```python
from langgraph.checkpoint.memory import MemorySaver

# 그래프 생성
g = StateGraph(State)

# 노드 등록
g.add_node("ingest_sentence", ingest_sentence)
g.add_node("dedup_gate", dedup_gate)
g.add_node("compress_context", compress_context)
g.add_node("summarize_rolling", summarize_rolling)
g.add_node("update_state", update_state)

# 에지 연결
g.add_edge(START, "ingest_sentence")
g.add_edge("ingest_sentence", "dedup_gate")

# 조건 분기: buffer>=6일 때만 요약 경로로
g.add_conditional_edges(
    "dedup_gate",
    should_summarize,                    # -> "do" | "skip"
    {"do": "compress_context", "skip": END}
)

g.add_edge("compress_context", "summarize_rolling")
g.add_edge("summarize_rolling", "update_state")
g.add_edge("update_state", END)

# 체크포인터(메모리) + 컴파일
memory = MemorySaver()
app = g.compile(checkpointer=memory)
```

포인트

add_conditional_edges(source, fn, mapping)에서 fn(state)는 문자 라벨을 반환, mapping의 키와 매칭됩니다.

루프를 만들고 싶으면 g.add_edge("update_state", "ingest_sentence") 같은 역방향 연결을 사용(스트리밍/폴링 모델일 때).

### 5) 실행/호출 패턴

- 단발 실행: 한 번의 입력에 대해 그래프를 한 바퀴 돌립니다.
- 스트리밍: 외부 이벤트(STT 문장 수신)마다 app.invoke()를 반복 호출하거나, 루프 에지를 구성해 상주형으로 운용합니다.

```python
# 초기 상태(필요 키만)
init_state: State = {
    "messages": [],
    "buffer_sentences": [], "running_summary": "",
    "last_overlap": [], "meta": {}
}

# 예: STT로 문장 3개 수신(아직 6개 미만이므로 요약 미실행)
state = app.invoke({**init_state, "buffer_sentences": ["문장1","문장2","문장3"]})

# 이후 추가로 3개 수신(총 6개 → 요약 경로 탑재)
state = app.invoke({**state, "buffer_sentences": state["buffer_sentences"] + ["문장4","문장5","문장6"]})

print(state["running_summary"])  # 최신 요약 확인
```
### 6) 중단/재개, 인터럽트 훅

- 중단/재개: checkpointer=MemorySaver()를 쓰면 각 스텝 후 스냅샷이 남아 장애 복구가 쉽습니다.
- 인터럽트 포인트: 특정 노드 전/후에 중단하고 외부 확인/개입 후 이어갈 수 있습니다.

```python
app = g.compile(
    checkpointer=memory,
    interrupt_before=["summarize_rolling"],   # 요약 직전에 멈춤(검토/수정/필터 가능)
    interrupt_after=[],
)
```

### 7) 서브그래프, 병렬, 재사용
(1) 서브그래프

복잡한 경로를 하나의 노드처럼 묶어 재사용할 수 있습니다.

```python
# 서브 상태/그래프 정의
class SubState(TypedDict):
    part: str

sub = StateGraph(SubState)
sub.add_node("a", lambda s: {"part": (s.get("part","") + "A")})
sub.add_node("b", lambda s: {"part": (s.get("part","") + "B")})
sub.add_edge(START, "a")
sub.add_edge("a", "b")
sub.add_edge("b", END)
sub_app = sub.compile()

# 메인 그래프에 서브그래프 삽입
g.add_node("subpipeline", sub_app)   # 함수처럼 사용
```

(2) 병렬(팬아웃) 패턴

여러 독립 후처리를 동시에 돌리고 마지막에 병합하는 패턴을 구성할 수 있습니다(노드들을 같은 선행 노드에서 이어 주고, 결과를 모으는 노드 하나로 합류).

### 8) 성능·안정화 포인트

- Partial Update 원칙: 노드는 바꿀 키만 반환하세요(불필요한 대용량 키 복사 방지).
- 토큰 예산 고정: running_summary는 항상 고정 길이로 유지해 지연이 시간에 따라 늘지 않게.
- 디바운스: 문장 이벤트 과밀 시 0.5~1.0s 지연 후 단일 트리거 처리.
- 로그/메트릭: meta에 prefill_ms, first_token_ms, decode_tps, n_in/n_out 저장 → 병목 진단.
- 루프/재진입 보호: 요약 중 재요약 진입 방지 플래그(예: meta["busy"]=True).

### 9) 시각화
- 구성이 복잡해지면 그래프를 그려보는 것이 유용합니다.

# 일부 환경에서 지원: g.get_graph().draw_png("graph.png") 또는 draw_dot
# (환경 의존, 문서/예제에 맞춰 사용)

### 10) 실무 템플릿(최소 뼈대)
```python
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

# 1) State 정의 (위 예시)
# 2) Node 함수들 정의 (ingest_sentence, dedup_gate, should_summarize, compress_context, summarize_rolling, update_state)

g = StateGraph(State)
for name, fn in [
    ("ingest_sentence", ingest_sentence),
    ("dedup_gate", dedup_gate),
    ("compress_context", compress_context),
    ("summarize_rolling", summarize_rolling),
    ("update_state", update_state),
]:
    g.add_node(name, fn)

g.add_edge(START, "ingest_sentence")
g.add_edge("ingest_sentence", "dedup_gate")
g.add_conditional_edges("dedup_gate", should_summarize, {"do": "compress_context", "skip": END})
g.add_edge("compress_context", "summarize_rolling")
g.add_edge("summarize_rolling", "update_state")
g.add_edge("update_state", END)

app = g.compile(checkpointer=MemorySaver())
```

이 템플릿 위에 LLM 호출 래퍼, vLLM 엔드포인트, 출력 형식 강제 프롬프트를 얹으면, 곧바로 STT 기반 슬라이딩 요약을 돌릴 수 있습니다.