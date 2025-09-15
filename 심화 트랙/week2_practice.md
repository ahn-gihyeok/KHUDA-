# 데이터분석 실무 100제 1장 - 기초 판다스 데이터 가공 요약

## 테크닉 001-003: 데이터 로딩 및 탐색

### 데이터 읽기
```python
# CSV 파일 읽기
item_master = pd.read_csv('item_master.csv')
item_master.head()

transaction_1 = pd.read_csv('transaction_1.csv')
transaction_1.head()

transaction_detail_1 = pd.read_csv('transaction_detail_1.csv')
transaction_detail_1.head()
```

**핵심 용어:**
- `pd.read_csv()`: CSV 파일을 DataFrame으로 읽어오는 함수
- `.head()`: 데이터프레임의 첫 5행을 출력

## 테크닉 004: 마스터 데이터 결합

### 데이터 결합
```python
join_data = pd.merge(left_data, customer_master, on='customer_id', how='left')
join_data = pd.merge(join_data, item_master, on='item_id', how='left')
join_data.head()
```

**핵심 용어:**
- `pd.merge()`: 두 DataFrame을 특정 키를 기준으로 결합
- `on`: 결합 기준이 되는 컬럼명
- `how`: 결합 방식 (left, right, inner, outer)

## 테크닉 005: 필요한 데이터 컬럼 만들기

### 컬럼 계산
```python
# quantity와 item_price를 곱해 price 컬럼 생성
quantity, item_price, price = 컬럼의 처음 5행을 출력해 분석
```

**핵심 용어:**
- 컬럼 간 연산: 새로운 컬럼 생성을 위한 기본 연산

## 테크닉 006: 데이터 정렬하기

### 데이터 정렬
```python
# 특정 컬럼 기준으로 정렬
sorted_data = data.sort_values('column_name')
```

## 테크닉 007: 결손치 파악하기

### 결손치 확인
```python
# 결손치 개수 확인
join_data.isnull().sum()

# 기술통계 확인
join_data.describe()
```

**핵심 용어:**
- `isnull()`: 결손치(NaN) 여부를 True/False로 반환
- `sum()`: 결손치 개수 계산 (True=1, False=0으로 계산)
- `describe()`: 기술통계량(count, mean, std, min, 25%, 50%, 75%, max) 출력

## 테크닉 008: 마스터 데이터의 조인 결과

### 조인 결과 확인
```python
# 1번에서는 customer_master, 2번에서는 item_master 조인
# 처음 5행의 출력 결과로 고객 정보와 상품 정보가 추가됨 확인
```

## 테크닉 009: 월별, 상품별로 데이터를 집계해보자

### 그룹화 및 집계
```python
# 월별, 상품별 집계
join_data.groupby(['payment_month', 'item_name']).sum()[['price', 'quantity']]
```

**핵심 용어:**
- `groupby()`: 특정 컬럼들을 기준으로 데이터 그룹화
- `sum()`: 그룹별 합계 계산
- `[['price', 'quantity']]`: 특정 컬럼만 선택

## 테크닉 010-012: 피벗 테이블

### 피벗 테이블 생성
```python
# 피벗 테이블로 월별, 상품별 집계 결과 표시
pd.pivot_table(join_data, 
               index='item_name', 
               columns='payment_month', 
               values=['price', 'quantity'], 
               aggfunc='sum')
```

**핵심 용어:**
- `pivot_table()`: 행과 열을 지정하여 교차표 형태로 데이터 재구성
- `index`: 행에 표시할 컬럼
- `columns`: 열에 표시할 컬럼  
- `values`: 집계할 값
- `aggfunc`: 집계 함수 (sum, mean, count 등)

## 주요 데이터 구조 정보

### 예시 데이터셋
- **item_master**: 상품 마스터 (item_id, item_name, item_price)
- **transaction_1**: 거래 데이터 (transaction_id, price, payment_date, customer_id)
- **transaction_detail_1**: 거래 상세 (detail_id, transaction_id, item_id, quantity)

### 결합 후 최종 데이터
- 고객 정보, 상품 정보, 거래 정보가 모두 포함된 통합 데이터
- 월별, 상품별 매출과 수량 분석 가능
- 피벗 테이블을 통한 시각적 데이터 확인

## 핵심 판다스 문법 정리

1. **데이터 읽기**: `pd.read_csv()`
2. **데이터 결합**: `pd.merge()`
3. **결손치 확인**: `isnull().sum()`
4. **기술통계**: `describe()`
5. **그룹화**: `groupby()`
6. **피벗테이블**: `pd.pivot_table()`
7. **데이터 미리보기**: `head()`