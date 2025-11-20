<!-- 9f142dd7-3de6-474d-944b-2ec0dcd26c9e 393d45a3-2210-4ce1-94f6-bf1697487c00 -->
# Pipeline Testing System Implementation

## Overview

각 파이프라인 스텝(Frame Extraction, Hold Detection, Wall Angle, Pose Estimation, Feature Extraction, Segmentation, Efficiency Scoring)의 입력/출력을 문서화하고, 개별 테스트를 위한 UI를 추가합니다.

## Implementation Steps

### 1. Create Testing Documentation (`/docs/TESTING_GUIDE.md`)

**파일 생성**: `/docs/TESTING_GUIDE.md`

**포함 내용**:

- 전체 파이프라인 스텝 개요 (7단계)
- 각 스텝별 상세 정보:
  - 입력 형식 및 요구사항
  - 출력 형식 및 스키마
  - 예상 결과 (성공/실패 케이스)
  - 독립 실행 명령어 및 코드 예제
  - 통합 테스트 시 의존성
  - 샘플 데이터 경로
  - 문제 해결 (Troubleshooting)
- 독립 테스트 vs 통합 테스트 가이드
- 테스트 데이터 준비 방법

**참고 파일**:

- `scripts/run_pipeline.py` - 전체 파이프라인 흐름
- `webapp/pipeline_runner.py` - 각 스텝 실행 로직
- `src/pose_ai/service/` - 각 서비스 모듈

### 2. Add Testing API Endpoints (`webapp/main.py`)

**수정 파일**: `/webapp/main.py`

**추가할 엔드포인트**:

```python
# Individual step testing endpoints
POST /api/test/extract-frames
POST /api/test/detect-holds
POST /api/test/estimate-wall-angle
POST /api/test/estimate-pose
POST /api/test/extract-features
POST /api/test/segment
POST /api/test/compute-efficiency

# Step validation endpoints
GET /api/test/validate/{step_name}
```

**각 엔드포인트 기능**:

- 해당 스텝만 독립적으로 실행
- 입력 검증 (필수 파일/파라미터 존재 확인)
- 출력 검증 (스키마 체크, 파일 생성 확인)
- 상세한 에러 메시지 및 디버깅 정보 반환

### 3. Create Testing UI Page

**새 파일**: `/webapp/templates/testing.html`

**UI 구성**:

- 상단: Step selector (7개 스텝 버튼/탭)
- 각 스텝별 섹션:
  - **Input Section**: 필요한 파일/파라미터 업로드/입력 폼
  - **Execute Button**: 해당 스텝만 실행
  - **Expected Output**: 기대되는 출력 형식 표시
  - **Actual Output**: 실행 결과 표시 (JSON viewer, 파일 다운로드 링크)
  - **Validation Results**: 출력 검증 결과 (✓/✗)
  - **Logs**: 실행 로그
- 하단: Full Pipeline Test 버튼 (통합 테스트)

**JavaScript 기능**:

- 각 스텝 독립 실행
- 결과 시각화 (JSON pretty-print, 이미지 미리보기)
- 파일 다운로드
- 이전 스텝 출력을 다음 스텝 입력으로 자동 연결

**새 파일**: `/webapp/static/testing.css` (선택적, 스타일 분리)

### 4. Update Navigation

**수정 파일**: `/webapp/templates/index.html`, `/webapp/templates/training.html`

**변경사항**:

- 네비게이션 바에 "Testing" 링크 추가
- 일관된 네비게이션 구조 유지

**추가 라우트**: `/webapp/main.py`

```python
@app.get("/testing", response_class=HTMLResponse)
async def testing_page(request: Request):
    return templates.TemplateResponse("testing.html", {"request": request})
```

### 5. Create Test Helper Module (선택적)

**새 파일**: `/src/pose_ai/testing/step_validator.py`

**기능**:

- 각 스텝의 입력/출력 스키마 정의
- 검증 함수 (validate_frame_extraction_output, etc.)
- 샘플 데이터 생성기
- 테스트 유틸리티

## Testing Strategy

### 독립 테스트 (Independent Testing)

- 각 스텝에 대한 샘플 입력 데이터 제공
- 스텝 단위로 실행 및 검증
- 빠른 디버깅 및 개발

### 통합 테스트 (Integrated Testing)

- 전체 파이프라인 실행
- 스텝 간 데이터 흐름 검증
- 엔드투엔드 테스트

## Files to Create/Modify

**새로 생성**:

- `/docs/TESTING_GUIDE.md`
- `/webapp/templates/testing.html`
- (선택적) `/src/pose_ai/testing/step_validator.py`

**수정**:

- `/webapp/main.py` - API 엔드포인트 및 라우트 추가
- `/webapp/templates/index.html` - 네비게이션 업데이트
- `/webapp/templates/training.html` - 네비게이션 업데이트

## Expected Outcomes

1. **문서화**: 개발자가 각 스텝의 입력/출력을 명확히 이해
2. **디버깅**: 문제가 있는 스텝을 빠르게 식별 및 수정
3. **개발 효율성**: 전체 파이프라인 실행 없이 개별 스텝 테스트
4. **품질 보증**: 각 스텝의 출력이 기대치를 만족하는지 자동 검증

### To-dos

- [ ] Create comprehensive testing guide documentation in /docs/TESTING_GUIDE.md
- [ ] Add individual step testing API endpoints to webapp/main.py
- [ ] Create testing.html UI page with step-by-step testing interface
- [ ] Update navigation in index.html and training.html to include Testing page
- [ ] Test all endpoints and UI functionality with sample data