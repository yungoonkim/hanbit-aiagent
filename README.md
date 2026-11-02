# 만들면서 배우는 AI 에이전트 개발 입문+실전 - 실습 코드 (비전공자도 이해할 수 있는 버전)

본 저장소는『만들면서 배우는 AI 에이전트 개발 입문+실전』도서의 공식 실습 코드 저장소([gongwon-nayeon/hanbit-aiagent](https://github.com/gongwon-nayeon/hanbit-aiagent))를 Fork하여, **프로그래밍 경험이 적은 비전공자도 이해할 수 있는 버전으로 학습 효율과 실습 편의성을 대폭 개선한 저장소**입니다.

---

## 🔄 원본 저장소(Fork 이전)와의 주요 차이점 안내

이 저장소는 원본 도서의 실습 흐름을 그대로 유지하면서, 비전공자분들이 겪기 쉬운 환경 설정 문제와 코드 독해의 어려움을 해결하기 위해 다음과 같은 변경 사항을 적용했습니다.

| 구분 | 원본 저장소 (`Before Fork`) | 현재 저장소 (`After Fork` - 비전공자도 이해할 수 있는 버전) |
| :--- | :--- | :--- |
| **1. 파이썬 환경 및 패키지 관리** | **Anaconda (`conda`)** 및 `pip` 중심<br>*(무거운 설치 용량, 가상환경 활성화 등 초보자에게 다소 복잡한 과정)* | **`uv` 단일 환경으로 전면 통일**<br>*(아나콘다 없이 `uv sync` 한 줄로 파이썬+가상환경+패키지 초고속 자동 설치 및 `uv run` 실행)* |
| **2. 소스 코드 주석 수준** | 최소한의 핵심 번호 표시(`# [ 1 ]`, `# [ 2 ]` 등) 위주 | **비전공자 눈높이의 한 줄 단위 상세 해설 주석 추가**<br>*(일상 비유와 파이썬 기초 문법 설명을 모든 `.py` 및 `.ipynb` 코드에 수록)* |
| **3. 챕터별 상세 학습 가이드** | 별도 웹 가이드 없음 (교재 본문 참조) | **`guides/` 폴더에 챕터별 인터랙티브 HTML 해설서 제공**<br>*(시각적 다이어그램, 용어 사전, 직접 눌러보는 그래프 시뮬레이터 포함)* |
| **4. 기본 LLM 모델** | OpenAI `gpt-4o` / `gpt-4o-mini` (`ChatOpenAI`) | • **PART 2**: Google **`gemini-3.5-flash-lite`**<br>• **CHAP 6 ~ CHAP 11**: Google **`gemini-3.8-flash`** (`ChatGoogleGenerativeAI` 및 Gemini OpenAI 호환 엔드포인트) 적용 |
| **5. 실행 호환성 개선** | MCP 실습 등에서 하위 프로세스 호출 시 시스템 `"python"` 직접 호출 | `uv run` 환경에서도 언제나 현재 가상환경(`.venv`)을 정확히 바라보도록 `sys.executable` 및 노트북 커널 설정 개선 |

---

## 목차

- [🔄 원본 저장소(Fork 이전)와의 주요 차이점 안내](#-원본-저장소fork-이전와의-주요-차이점-안내)
- [파일 구성](#파일-구성)
  - [guides: 비전공자 맞춤형 챕터별 학습 가이드 (HTML)](#guides-비전공자-맞춤형-챕터별-학습-가이드-html)
  - [PART2 (4-5장): 랭그래프 기초](#part2-4-5장-랭그래프-기초)
  - [CHAP6_single-agent: 싱글 에이전트](#chap6_single-agent-싱글-에이전트)
  - [CHAP7_multi-agent: 멀티 에이전트 시스템](#chap7_multi-agent-멀티-에이전트-시스템)
  - [CHAP8_memory: 메모리 관리](#chap8_memory-메모리-관리)
  - [CHAP9_MCP: Model Context Protocol](#chap9_mcp-model-context-protocol)
  - [CHAP10_A2A: Agent-to-Agent 프로토콜](#chap10_a2a-agent-to-agent-프로토콜)
  - [CHAP11_final-project: 실전형 멀티 에이전트 시스템](#chap11_final-project-실전형-멀티-에이전트-시스템)
- [환경 설정 및 실행 방법 (uv 기반)](#환경-설정-및-실행-방법-uv-기반)
  - [1. uv 설치하기](#1-uv-설치하기)
  - [2. 저장소 클론 및 폴더 이동](#2-저장소-클론-및-폴더-이동)
  - [3. 가상환경 생성 및 패키지 일괄 설치 (uv sync)](#3-가상환경-생성-및-패키지-일괄-설치-uv-sync)
  - [4. 환경 변수(.env) 설정하기](#4-환경-변수env-설정하기)
  - [5. 실습 코드 및 주피터 노트북 실행하기](#5-실습-코드-및-주피터-노트북-실행하기)
  - [6. LangGraph Studio 실행하기](#6-langgraph-studio-실행하기)
- [문의](#문의)

---

## 파일 구성

본 저장소는 책의 흐름에 따라 구성되어 있으며, 각 챕터별로 독립된 실습 예제와 상세 해설 가이드를 제공합니다.

### guides: 비전공자 맞춤형 챕터별 학습 가이드 (HTML)
```
guides/
├── index.html            # 전체 챕터 가이드 메인 목차 (브라우저로 열기)
├── PART2_guide.html      # PART 2 (4~5장): 랭그래프 기초 해설서
├── CHAP6_guide.html      # CHAP 6: 싱글 에이전트 (웹검색, 코딩, 미들웨어, Self-RAG) 해설서
├── CHAP7_guide.html      # CHAP 7: 멀티 에이전트 (Network, Supervisor, Planning) 해설서
├── CHAP8_guide.html      # CHAP 8: 메모리 관리 (단기 기억 vs 장기 기억) 해설서
├── CHAP9_guide.html      # CHAP 9: MCP (Model Context Protocol) 해설서
├── CHAP10_guide.html     # CHAP 10: A2A (Agent-to-Agent 프로토콜) 해설서
└── CHAP11_guide.html     # CHAP 11: 실전 파이널 프로젝트 통합 아키텍처 해설서
```

### PART2 (4-5장): 랭그래프 기초
```
PART2/
├── 4.4_main.py
├── 5.2_랭그래프 기본 개념 이해하기.ipynb
└── 5.3_랭그래프로 에이전트 설계하고 구현하기.ipynb
```

### CHAP6_single-agent: 싱글 에이전트
```
CHAP6_single-agent/
├── coding_agent/          # 6.3 코딩 에이전트 만들기
├── rag_agent/             # 6.5 RAG를 위한 에이전트 만들기
├── web_agent/             # 6.2 웹 검색 에이전트 만들기
├── create_agent/          # 6.4 create_agent 상세 구조 이해하기
└── langgraph.json
```

### CHAP7_multi-agent: 멀티 에이전트 시스템
```
CHAP7_multi-agent/
├── supervisor_agent_web/        # 7.4 웹페이지를 요약해서 데이터베이스에 저장하는 에이전트
├── supervisor_agent_triple/     # 7.5 최신 문서 검색 + 내부 DB 검색 + 템플릿 답변 3중 멀티 에이전트
├── supervisor_planning_agent/   # 7.6 자료 조사 전문가 + 문서 작성 전문가 에이전트
├── network_agent/               # 7.3 정보 검색을 기반으로 차트를 그려주는 에이전트
├── how_to_use_command.ipynb
├── outputs/
└── langgraph.json
```

### CHAP8_memory: 메모리 관리
```
CHAP8_memory/
├── short-term memory.ipynb
└── long-term memory.ipynb
```

### CHAP9_MCP: Model Context Protocol
```
CHAP9_MCP/
├── mcp_agent/              # 9.3, 9.4 MCP 서버/클라이언트 기본 실습
└── mcp_multi_agent/        # 9.5 랭그래프에서 MCP 기반 멀티 에이전트 구현하기
```

### CHAP10_A2A: Agent-to-Agent 프로토콜
```
CHAP10_A2A/
├── hello_world/           # 10.3 A2A의 기본 사용법 이해하기
└── multi_agent/           # 10.4 MCP와 A2A를 활용한 멀티 에이전트 구축하기
```

### CHAP11_final-project: 실전형 멀티 에이전트 시스템
```
CHAP11_final-project/        # [PART 5] 실전 멀티 에이전트 프로젝트
├── orchestrator_agent/
├── web_research_agent/
├── internal_rag_agent/
├── file_management_agent/
├── common/
├── test_client.py
└── README.md
```

---

## 환경 설정 및 실행 방법 (uv 기반)

무겁고 복잡한 Anaconda(아나콘다)를 설치할 필요 없이, **`uv`** 하나만 설치하면 파이썬 버전(Python 3.11 이상) 설치부터 가상환경(`.venv`) 생성, 패키지 설치까지 단 한 줄의 명령어로 해결됩니다!

### 1. uv 설치하기

운영체제에 맞는 터미널(macOS는 `터미널`, Windows는 `PowerShell`)을 열고 아래 명령어를 복사해 실행합니다.

```bash
# macOS / Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows (PowerShell)
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

설치 후 터미널을 껐다가 다시 켜고, 아래 명령어로 설치가 잘 되었는지 확인합니다.
```bash
uv --version
```

### 2. 저장소 클론 및 폴더 이동

```bash
git clone https://github.com/gongwon-nayeon/hanbit-aiagent.git
cd hanbit-aiagent
```

### 3. 가상환경 생성 및 패키지 일괄 설치 (`uv sync`)

프로젝트 루트 폴더(`hanbit-aiagent`)에서 아래 명령어 **단 한 줄**만 실행하면 됩니다!
- 컴퓨터에 Python 3.11+가 없어도 `uv`가 자동으로 파이썬을 내려받습니다.
- `.venv` 가상환경 폴더를 자동으로 만들고, `pyproject.toml`에 적힌 모든 필수 패키지(LangGraph, LangChain, Gemini/OpenAI, MCP, A2A, Jupyter 커널 등)를 초고속으로 설치합니다.

```bash
uv sync
```

> 💡 **가상환경 직접 활성화하기 (선택 사항)**
> `uv run <명령어>`를 사용하면 가상환경을 매번 활성화(`activate`)하지 않아도 자동으로 `.venv` 환경에서 실행됩니다. 만약 터미널에서 가상환경을 직접 켜고 싶다면 아래 명령어를 사용하세요.
> ```bash
> source .venv/bin/activate  # macOS / Linux
> .venv\Scripts\activate     # Windows
> ```

### 4. 환경 변수(`.env`) 설정하기

프로젝트 루트 디렉터리(또는 실행할 각 챕터 폴더)에 `.env` 파일을 생성하고 발급받은 API 키를 입력합니다.

```bash
# .env 파일 예시
GOOGLE_API_KEY=your_google_gemini_api_key
OPENAI_API_KEY=your_openai_api_key
TAVILY_API_KEY=your_tavily_api_key
```

### 5. 실습 코드 및 주피터 노트북 실행하기

#### ① 파이썬(`.py`) 파일 실행하기
명령어 앞에 **`uv run`** 만 붙이면 언제 어디서든 가상환경(`.venv`)의 파이썬으로 깔끔하게 실행됩니다.

```bash
# 예시 1: PART2의 첫 LLM 호출 예제 실행
uv run python PART2/4.4_main.py

# 예시 2: CHAP9의 MCP 에이전트 클라이언트 실행
cd CHAP9_MCP/mcp_agent
uv run python client.py
```

#### ② 주피터 노트북(`.ipynb`) 실행하기 (VS Code / Cursor / Antigravity)
1. 에디터에서 `.ipynb` 파일(예: `PART2/5.2_랭그래프 기본 개념 이해하기.ipynb`)을 엽니다.
2. 우측 상단의 **[커널 선택(Select Kernel)]** 버튼을 클릭합니다.
3. **[Python Environments...]** → **`.venv (Python 3.x.x)`** 를 선택하면 바로 셀을 실행할 수 있습니다!

### 6. LangGraph Studio 실행하기

`uv run langgraph dev` 명령어를 사용하여 LangGraph Studio에서 에이전트를 시각적으로 확인하고 테스트할 수 있습니다. (자세한 설명은 책의 **6.2.5절** 참고)

#### `langgraph.json` 기본 구조
```json
{
  "dependencies": ["./agent_directory"],
  "graphs": {
    "agent": "./agent_directory/agent.py:graph"
  },
  "env": ".env"
}
```

#### 실행 방법
```bash
# 해당 챕터 디렉터리로 이동 (예: CHAP6 또는 CHAP7)
cd CHAP6_single-agent

# uv를 통해 LangGraph Studio 개발 서버 실행
uv run langgraph dev
```

---

## 문의

### GitHub Issues
코드 실행 중 문제가 발생하거나 질문이 있으시면 GitHub Issues를 활용해주세요. 질의응답 커뮤니티로 활용하실 수 있습니다.

[이슈 등록하기](https://github.com/gongwon-nayeon/hanbit-aiagent/issues)

### 저자 연락처
- 이메일: uoahvu@gmail.com
- 유튜브 채널: [**공원나연**](https://www.youtube.com/@gongwon-nayeon)
