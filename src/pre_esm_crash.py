import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib  # 스케일러 저장을 위해 추가
import re

def load_data(call119_path, cat119_path):
    encodings_to_try = ['utf-8', 'utf-8-sig', 'cp949', 'euc-kr']

    def read_csv_robust(path):
        for encoding in encodings_to_try:
            try:
                df = pd.read_csv(path, encoding=encoding, index_col=False)
                print(f"✅ '{os.path.basename(path)}' 파일을 '{encoding}' 인코딩으로 읽기 성공.")
                return df
            except (UnicodeDecodeError, pd.errors.ParserError):
                continue
        raise Exception(f"❌ '{os.path.basename(path)}' 파일을 다음 인코딩으로 읽을 수 없습니다: {encodings_to_try}")

    print(f"\n--- 데이터 파일 로드 시작 ---")
    call119_df = read_csv_robust(call119_path)
    cat119_df = read_csv_robust(cat119_path)

    print(f"call119 데이터 크기: {call119_df.shape}")
    print(f"cat119 데이터 크기: {cat119_df.shape}")

    return call119_df, cat119_df

def preprocess_people_data(people_raw_path):
    print("인구 데이터 전처리 시작...")

    # 멀티헤더 읽기
    df_raw = pd.read_csv(people_raw_path, header=[0, 1])

    # 멀티인덱스 -> 단일 컬럼으로 flatten
    df_raw.columns = [
        f"{upper}_{lower}".strip() if lower != '' and lower != 'Unnamed: 1_level_1' else upper
        for upper, lower in df_raw.columns
    ]

    # 데이터 파트만 추출 (index 초기화)
    df_raw.reset_index(drop=True, inplace=True)

    # 합계 / 남 / 여 순서대로 존재한다고 가정
    total_row = df_raw.iloc[0]
    male_row = df_raw.iloc[1]
    female_row = df_raw.iloc[2]

    people_data = []
    for col in df_raw.columns[2:]:
        gu, dong = col.split('_')
        total = int(total_row[col])
        male = int(male_row[col])
        female = int(female_row[col])
        people_data.append([gu, dong, total, male, female])

    df_people = pd.DataFrame(people_data, columns=[
        'address_gu', 'sub_address', 'population_total', 'population_male', 'population_female'
    ])

    print(f"✅ 인구 데이터 변환 완료: {df_people.shape}")
    if '강서구' in df_people['address_gu'].unique():
        print("    -> '강서구' 데이터가 정상적으로 포함되었습니다.")
    else:
        print("    -> ⚠️ 경고: 처리 후에도 '강서구' 데이터가 없습니다. 원본 CSV 파일을 확인해주세요.")

    return df_people

def create_dong_aggregation_mapping(people_df):
    """
    동 이름 패턴을 분석해서 자동으로 통합 매핑을 생성
    """
    dong_mapping = {}

    # 1. 기본 동 이름 추출 (숫자 제거)
    people_df_copy = people_df.copy()
    people_df_copy['base_dong'] = people_df_copy['sub_address'].str.replace(r'\d+동$', '동', regex=True)

    # 2. 각 기본 동별로 그룹화
    base_dong_groups = people_df_copy.groupby(['address_gu', 'base_dong'])

    for (gu, base_dong), group in base_dong_groups:
        if len(group) > 1:
            # 합계 계산
            total_pop = group['population_total'].sum()
            male_pop = group['population_male'].sum()
            female_pop = group['population_female'].sum()

            # 통합 동 이름으로 매핑
            dong_mapping[f"{gu}_{base_dong}"] = {
                'population_total': total_pop,
                'population_male': male_pop,
                'population_female': female_pop,
                'source_dongs': group['sub_address'].tolist()
            }

            print(f"통합 동 생성: {gu} {base_dong} = {group['sub_address'].tolist()}")

    return dong_mapping


def merge_people_data(merged_df, people_df):

    print("향상된 인구 데이터 병합 중...")

    # 1. 동 통합 매핑 생성
    dong_mapping = create_dong_aggregation_mapping(people_df)

    # 2. 병합키 전처리 (소문자, 공백 제거)
    merged_df_copy = merged_df.copy()
    people_df_copy = people_df.copy()

    for col in ['address_gu', 'sub_address']:
        merged_df_copy[col] = merged_df_copy[col].astype(str).str.strip().str.lower()
        people_df_copy[col] = people_df_copy[col].astype(str).str.strip().str.lower()

    # 3. 기본 병합 시도 (정확 매칭 - 대저1동 = 대저1동)
    merged_result = pd.merge(
        merged_df_copy,
        people_df_copy[['address_gu', 'sub_address', 'population_total', 'population_male', 'population_female']],
        on=['address_gu', 'sub_address'],
        how='left'
    )

    # 4. 매칭 안 된 행들에 대해 통합 동 매핑 적용
    unmatched_mask = merged_result['population_total'].isna()

    print(f"직접 매칭 성공: {(~unmatched_mask).sum()}개")
    print(f"직접 매칭 실패: {unmatched_mask.sum()}개")

    for idx in merged_result[unmatched_mask].index:
        gu = merged_result.loc[idx, 'address_gu']
        dong = merged_result.loc[idx, 'sub_address']

        # 기본 동 이름 생성 (숫자 제거) - 동대신동 <- 동대신1동+동대신2동+동대신3동
        base_dong = re.sub(r'\d+동$', '동', dong)
        mapping_key = f"{gu}_{base_dong}"

        if mapping_key in dong_mapping:
            # 통합 동 데이터 사용
            merged_result.loc[idx, 'population_total'] = dong_mapping[mapping_key]['population_total']
            merged_result.loc[idx, 'population_male'] = dong_mapping[mapping_key]['population_male']
            merged_result.loc[idx, 'population_female'] = dong_mapping[mapping_key]['population_female']

            source_dongs = dong_mapping[mapping_key]['source_dongs']
          #  print(f"통합 동 매핑 적용: {gu} {dong} <- {source_dongs}")

    # 5. 여전히 매칭 안 된 경우 구별 최소값 사용
    still_unmatched = merged_result['population_total'].isna()

    if still_unmatched.sum() > 0:
        print(f"구별 최소값으로 처리할 동: {still_unmatched.sum()}개")

        # 구별 최소값 계산 - 더 안정적인 방법 사용
        gu_min_stats = people_df_copy.groupby('address_gu').agg({
            'population_total': 'min',
            'population_male': 'min',
            'population_female': 'min'
        })

        for idx in merged_result[still_unmatched].index:
            gu = merged_result.loc[idx, 'address_gu']
            dong = merged_result.loc[idx, 'sub_address']

            if gu in gu_min_stats.index:
                merged_result.loc[idx, 'population_total'] = gu_min_stats.loc[gu, 'population_total']
                merged_result.loc[idx, 'population_male'] = gu_min_stats.loc[gu, 'population_male']
                merged_result.loc[idx, 'population_female'] = gu_min_stats.loc[gu, 'population_female']

                print(f"구별 최소값 적용: {gu} {dong} -> {gu_min_stats.loc[gu, 'population_total']}명")
            else:
                # 구 정보도 없는 경우 전체 최소값 사용
                min_total = people_df_copy['population_total'].min()
                min_male = people_df_copy['population_male'].min()
                min_female = people_df_copy['population_female'].min()

                merged_result.loc[idx, 'population_total'] = min_total
                merged_result.loc[idx, 'population_male'] = min_male
                merged_result.loc[idx, 'population_female'] = min_female

                print(f"전체 최소값 적용: {gu} {dong} -> {min_total}명")

    # 6. 최종 결과 확인
    final_unmatched = merged_result['population_total'].isna().sum()
    if final_unmatched > 0:
        print(f"⚠️  최종적으로 매칭되지 않은 행: {final_unmatched}개")
    else:
        print("✅ 모든 행의 인구 데이터 매칭 완료")

    # 7. 원본 데이터프레임의 인덱스와 구조 유지하면서 인구 데이터만 추가
    merged_df_result = merged_df.copy()
    merged_df_result['population_total'] = merged_result['population_total']
    merged_df_result['population_male'] = merged_result['population_male']
    merged_df_result['population_female'] = merged_result['population_female']

    print(f"✅ 향상된 인구 데이터 병합 완료: {merged_df_result.shape}")

    return merged_df_result


def preprocess_traffic_data(traffic_raw_path, year):
    """
    교통사고 통계 원본 데이터를 전처리하고, 모델 학습에 사용할 피처를 생성합니다.
    """
    print(f"교통사고 데이터 전처리 시작 ({year}년)...")

    try:
        # 다양한 인코딩으로 파일 로드 시도
        encodings_to_try = ['utf-8', 'utf-8-sig', 'cp949', 'euc-kr']
        df_raw = None
        for encoding in encodings_to_try:
            try:
                df_raw = pd.read_csv(traffic_raw_path, encoding=encoding)
                print(f"✅ '{os.path.basename(traffic_raw_path)}' 파일 로드 성공 (인코딩: {encoding})")
                break
            except (UnicodeDecodeError, pd.errors.ParserError):
                continue
        if df_raw is None:
            raise Exception("모든 인코딩으로 파일 읽기에 실패했습니다.")

    except FileNotFoundError:
        print(f"❌ 오류: 교통사고 데이터 파일을 찾을 수 없습니다: {traffic_raw_path}")
        return pd.DataFrame()

    metrics_map = {
        '사고[건]': 'accident_count',
        '사망[명]': 'fatality_count',
        '부상[명]': 'injury_count',
        '(중상자[명])': 'serious_injury_count'
    }

    processed_dfs = []
    day_cols = [f'{i:02d}일' for i in range(1, 32)]

    for metric_kr, metric_en in metrics_map.items():
        if '사고일' not in df_raw.columns:
            print(f"❌ 오류: '{os.path.basename(traffic_raw_path)}' 파일에 '사고일' 컬럼이 없습니다.")
            return pd.DataFrame()

        df_metric = df_raw[df_raw['사고일'] == metric_kr].copy()
        df_melted = df_metric.melt(id_vars=['시군구', '사고월'], value_vars=day_cols, var_name='일', value_name=metric_en)
        df_melted[metric_en] = pd.to_numeric(df_melted[metric_en].replace('-', '0'), errors='coerce').fillna(0).astype(
            int)
        processed_dfs.append(df_melted)

    from functools import reduce
    df_merged = reduce(lambda left, right: pd.merge(left, right, on=['시군구', '사고월', '일']), processed_dfs)

    df_merged['사고월'] = df_merged['사고월'].str.replace('월', '').astype(int)
    df_merged['일'] = df_merged['일'].str.replace('일', '').astype(int)

    date_str = str(year) + '-' + df_merged['사고월'].astype(str) + '-' + df_merged['일'].astype(str)
    df_merged['tm_dt'] = pd.to_datetime(date_str, errors='coerce')

    df_merged.dropna(subset=['tm_dt'], inplace=True)
    df_merged['tm'] = df_merged['tm_dt'].dt.strftime('%Y%m%d')

    df_merged.rename(columns={'시군구': 'address_gu'}, inplace=True)
    df_merged['total_casualties'] = df_merged['fatality_count'] + df_merged['injury_count']

    final_features = [
        'tm', 'address_gu',
        'accident_count',
        'total_casualties'
    ]
    df_final = df_merged[final_features]

    print(f"✅ {year}년 교통사고 데이터 처리 완료 (기본 피처만 사용): {df_final.shape}")
    return df_final

def preprocess_weather_alerts(df):
    """
    기상 특보 데이터를 전처리하고 날짜별로 집계합니다.
    """
    print("기상 특보 데이터 전처리 중...")

    if df.empty:
        print("❌ 기상 특보 데이터가 비어있습니다.")
        return pd.DataFrame()

    # 컬럼 이름 후보 확장 (더 다양한 경우의 수 고려)
    possible_cols = [
        'warning_type', '특보종류', 'alert_type', '특보내용',
        'warning', 'alert', 'type', '종류', '내용'
    ]

    alert_col_name = None
    for col in possible_cols:
        if col in df.columns:
            alert_col_name = col
            print(f"✅ 기상 특보 컬럼으로 '{alert_col_name}'을 사용합니다.")
            break

    if alert_col_name is None:
        print(f"❌ 기상 특보 유형 컬럼을 찾을 수 없습니다.")
        print(f"   실제 컬럼: {list(df.columns)}")
        return pd.DataFrame()

    try:
        # 1. 날짜 컬럼 처리 (더 유연하게)
        df = df.copy()
        df['tm'] = pd.to_datetime(df['tm'], errors='coerce')

        # 날짜 변환 실패한 행 제거
        invalid_dates = df['tm'].isna().sum()
        if invalid_dates > 0:
            print(f"⚠️  날짜 변환 실패한 행 {invalid_dates}개 제거")
            df = df.dropna(subset=['tm'])

        if df.empty:
            print("❌ 유효한 날짜 데이터가 없습니다.")
            return pd.DataFrame()

        # 2. 특보 데이터 정리 및 분리
        # NaN 값 처리
        df[alert_col_name] = df[alert_col_name].fillna('')

        # 문자열 정리: 주의보/경보 제거, 공백 정리
        df['cleaned_alerts'] = (df[alert_col_name]
                                .str.replace('주의보|경보', '', regex=True)
                                .str.replace(r'\s+', ' ', regex=True)  # 연속된 공백 제거
                                .str.strip())

        # 3. 쉼표로 분리된 특보들을 개별 행으로 분할
        df_expanded = df.assign(
            alert_type=df['cleaned_alerts'].str.split(',')
        ).explode('alert_type')

        # 빈 값이나 공백만 있는 특보명 제거
        df_expanded = df_expanded[
            df_expanded['alert_type'].str.strip().str.len() > 0
            ]
        df_expanded['alert_type'] = df_expanded['alert_type'].str.strip()

        # 4. 원-핫 인코딩
        alert_dummies = pd.get_dummies(df_expanded['alert_type'], prefix='alert')

        # 5. 날짜와 결합하여 집계
        alert_processed = pd.concat([
            df_expanded[['tm']].reset_index(drop=True),
            alert_dummies.reset_index(drop=True)
        ], axis=1)

        # 6. 날짜별로 집계 (특보가 하루에 여러 번 나와도 1로 처리)
        alert_agg = alert_processed.groupby('tm').max().reset_index()

        # 7. 결과 검증
        alert_features = [col for col in alert_agg.columns if col.startswith('alert_')]
        print(f"✅ 기상 특보 데이터 전처리 완료:")
        print(f"   - 처리된 날짜 범위: {alert_agg['tm'].min()} ~ {alert_agg['tm'].max()}")
        print(f"   - 생성된 특보 피처: {len(alert_features)}개")
        print(f"   - 특보 종류: {[col.replace('alert_', '') for col in alert_features]}")

        return alert_agg

    except Exception as e:
        print(f"❌ 기상 특보 데이터 전처리 중 오류 발생: {e}")
        return pd.DataFrame()


def merge_weather_alerts(merged_df, alert_agg_df):
    """
    기상 특보 데이터를 메인 데이터프레임에 병합하는 별도 함수
    """
    print("기상 특보 데이터 병합 중...")

    if alert_agg_df.empty:
        print("⚠️  기상 특보 데이터가 없어 병합을 건너뜁니다.")
        return merged_df

    # 날짜 컬럼 타입 통일
    merged_df['tm'] = pd.to_datetime(merged_df['tm'], format='%Y%m%d', errors='coerce')
    alert_agg_df['tm'] = pd.to_datetime(alert_agg_df['tm'], errors='coerce')

    # 병합 전 데이터 확인
    print(f"   - 메인 데이터 날짜 범위: {merged_df['tm'].min()} ~ {merged_df['tm'].max()}")
    print(f"   - 특보 데이터 날짜 범위: {alert_agg_df['tm'].min()} ~ {alert_agg_df['tm'].max()}")

    # Left Join으로 병합
    result_df = pd.merge(merged_df, alert_agg_df, on='tm', how='left')

    # 특보 컬럼들의 NaN을 0으로 채움
    alert_cols = [col for col in result_df.columns if col.startswith('alert_')]
    result_df[alert_cols] = result_df[alert_cols].fillna(0).astype(int)

    # 병합 결과 확인
    total_alerts = result_df[alert_cols].sum().sum()
    coverage = (result_df[alert_cols].sum(axis=1) > 0).mean()

    print(f"✅ 기상 특보 데이터 병합 완료:")
    print(f"   - 특보 피처 수: {len(alert_cols)}개")
    print(f"   - 전체 특보 발생 횟수: {total_alerts}회")
    print(f"   - 특보가 있는 날의 비율: {coverage:.1%}")

    return result_df


def clean_column_names(df):
    """컬럼명 정리 함수"""
    new_columns = []
    for col in df.columns:
        col_clean = col.lower()
        if '.' in col_clean:
            col_clean = col_clean.split('.')[-1]
        new_columns.append(col_clean)
    df.columns = new_columns
    return df


def aggregate_cat119_by_location_time(cat119_df):
    """cat119 데이터를 지역별, 시간별로 집계"""
    print("cat119 데이터 집계 중...")
    cat_dummies = pd.get_dummies(cat119_df['cat'], prefix='cat')
    subcat_dummies = pd.get_dummies(cat119_df['sub_cat'], prefix='subcat')
    cat119_expanded = pd.concat([
        cat119_df[['tm', 'address_city', 'address_gu', 'sub_address', 'stn']],
        cat_dummies,
        subcat_dummies,
        cat119_df[['call_count']]
    ], axis=1)
    groupby_cols = ['tm', 'address_city', 'address_gu', 'sub_address', 'stn']
    numeric_cols = cat_dummies.columns.tolist() + subcat_dummies.columns.tolist() + ['call_count']
    agg_dict = {col: 'sum' for col in numeric_cols}
    cat119_agg = cat119_expanded.groupby(groupby_cols).agg(agg_dict).reset_index()
    print(f"cat119 집계 완료: {cat119_agg.shape}")
    print(f"생성된 신고 유형 피처 수: {len(cat_dummies.columns) + len(subcat_dummies.columns)}")
    return cat119_agg


def merge_call_cat_data(call119_df, cat119_agg):
    """call119와 집계된 cat119 데이터 병합"""
    print("call119와 cat119 데이터 병합 중...")
    merge_cols = ['tm', 'address_city', 'address_gu', 'sub_address', 'stn']
    for col in merge_cols:
        if col in call119_df.columns and col in cat119_agg.columns:
            if call119_df[col].dtype == 'object':
                call119_df[col] = call119_df[col].astype(str).str.strip()
                cat119_agg[col] = cat119_agg[col].astype(str).str.strip()
    merged_df = call119_df.merge(cat119_agg, on=merge_cols, how='left', suffixes=('', '_cat'))
    cat_cols = [col for col in cat119_agg.columns if col.startswith(('cat_', 'subcat_'))]
    merged_df[cat_cols] = merged_df[cat_cols].fillna(0)
    if 'call_count_cat' in merged_df.columns:
        merged_df['call_count'] = merged_df['call_count_cat'].fillna(merged_df['call_count'])
        merged_df.drop(columns=['call_count_cat'], inplace=True)
    print(f"병합 완료: {merged_df.shape}")
    print(f"병합 성공률: {(merged_df[cat_cols].sum(axis=1) > 0).mean():.2%}")
    return merged_df


def handle_missing_values(df):
    """결측치 처리"""
    print("결측치 처리 중...")
    weather_cols = ['ta_max', 'ta_min', 'ta_max_min', 'hm_min', 'hm_max', 'ws_max', 'ws_ins_max', 'rn_day']
    for col in weather_cols:
        if col in df.columns:
            df[col] = df[col].replace(-99.0, np.nan)
    missing_counts = df.isnull().sum()
    if missing_counts.sum() > 0:
        print("결측치 현황:")
        for col, count in missing_counts[missing_counts > 0].items():
            print(f"  {col}: {count}개 ({count / len(df) * 100:.1f}%)")
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if df[col].isnull().sum() > 0:
                if 'address_gu' in df.columns:
                    df[col] = df.groupby('address_gu')[col].transform(lambda x: x.fillna(x.mean()))
                df[col] = df[col].fillna(df[col].mean())
    return df


def create_time_features(df, time_col='tm'):
    """시간 관련 피처 생성"""
    print("시간 관련 피처 생성 중...")
    df[time_col] = pd.to_datetime(df[time_col], errors='coerce')
    df['year'] = df[time_col].dt.year
    df['month'] = df[time_col].dt.month
    df['day'] = df[time_col].dt.day
    df['dayofweek'] = df[time_col].dt.dayofweek
    df['quarter'] = df[time_col].dt.quarter
    df['is_weekend'] = (df['dayofweek'] >= 5).astype(int)
    df['season'] = df['month'].map({
        12: 'winter', 1: 'winter', 2: 'winter',
        3: 'spring', 4: 'spring', 5: 'spring',
        6: 'summer', 7: 'summer', 8: 'summer',
        9: 'autumn', 10: 'autumn', 11: 'autumn'
    })
    df['is_holiday'] = 0
    print("시간 피처 생성 완료")
    return df


def preprocess_data(call119_df, cat119_df, weather_alert_df, traffic_df,people_df=None):
    """개선된 메인 전처리 함수"""
    print("\n--- 데이터 전처리 시작 ---")

    # 1. 컬럼명 정리
    call119_df = clean_column_names(call119_df.copy())
    cat119_df = clean_column_names(cat119_df.copy())
    weather_alert_df = clean_column_names(weather_alert_df.copy())

    # 2. 불필요한 컬럼 제거
    for df in [call119_df, cat119_df]:
        if 'unnamed: 0' in df.columns:
            df.drop(columns=['unnamed: 0'], inplace=True)

    # 3. call119와 cat119 병합
    if not cat119_df.empty:
        cat119_agg = aggregate_cat119_by_location_time(cat119_df)
        merged_df = merge_call_cat_data(call119_df, cat119_agg)
    else:
        merged_df = call119_df.copy()

    # 4. 기상 특보 데이터 전처리 및 병합
    if not weather_alert_df.empty:
        alert_agg_df = preprocess_weather_alerts(weather_alert_df)
        merged_df = merge_weather_alerts(merged_df, alert_agg_df)
    else:
        print("⚠️  기상 특보 데이터가 없습니다.")
        # 5. 교통사고 데이터 병합
    if traffic_df is not None and not traffic_df.empty:
        print("교통사고 데이터 병합 중...")
        # tm 컬럼 타입을 병합 전에 통일 (YYYYMMDD 형식의 문자열로)
        merged_df['tm'] = merged_df['tm'].astype(str).str.replace('-', '')
        traffic_df['tm'] = traffic_df['tm'].astype(str)

        # 'address_gu'의 데이터 타입과 공백 등을 통일
        merged_df['address_gu'] = merged_df['address_gu'].astype(str).str.strip()
        traffic_df['address_gu'] = traffic_df['address_gu'].astype(str).str.strip()

        merged_df = pd.merge(merged_df, traffic_df, on=['tm', 'address_gu'], how='left')

        # 병합 후 생성된 NaN은 0으로 채움 (사고가 없었던 날)
        traffic_cols = ['accident_count', 'total_casualties', 'injury_per_accident', 'accident_moving_average_3days']
        for col in traffic_cols:
            if col in merged_df.columns:
                merged_df[col] = merged_df[col].fillna(0)
        print(f"✅ 교통사고 데이터 병합 완료. 현재 데이터 크기: {merged_df.shape}")
    else:
        print("⚠️  교통사고 데이터가 없어 병합을 건너뜁니다.")

    import sys
    print("\n[디버깅] 교통사고 데이터 병합 후 'address_gu' 목록 확인")



    if people_df is not None:
        merged_df = merge_people_data(merged_df, people_df)


    # 5. 나머지 전처리 과정
    merged_df = handle_missing_values(merged_df)
    merged_df = create_time_features(merged_df, 'tm')

    print("데이터 전처리 완료.")
    print(f"최종 데이터 크기: {merged_df.shape}")

    return merged_df


def encode_features(df):

    print("피처 인코딩 시작...")
    final_df = df.copy()
    categorical_cols = [
        'address_city', 'address_gu', 'sub_address',
        'year', 'month', 'day', 'dayofweek', 'quarter', 'season'
    ]
    existing_categorical_cols = [col for col in categorical_cols if col in final_df.columns]

    if existing_categorical_cols:

        final_df = pd.get_dummies(final_df, columns=existing_categorical_cols, drop_first=False)
        print(f"One-hot 인코딩 완료: {len(existing_categorical_cols)}개 범주형 변수")

    cat_features = [col for col in final_df.columns if col.startswith(('cat_', 'subcat_'))]
    print(f"신고 유형 피처: {len(cat_features)}개")
    print("피처 인코딩 완료.")
    print(f"최종 피처 수: {final_df.shape[1]}")
    return final_df


def split_data_time_series(df, target_column='call_count', train_ratio=0.8, val_ratio=0.1):
    """시계열 데이터 분할 및 스케일링"""
    print("시계열 데이터 분할 및 스케일링 시작...")
    time_col = 'tm'
    df_sorted = df.sort_values(by=time_col).reset_index(drop=True)

    n = len(df_sorted)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    train_df = df_sorted.iloc[:train_end]
    val_df = df_sorted.iloc[train_end:val_end]
    test_df = df_sorted.iloc[val_end:]

    cols_to_drop = [target_column, 'ID', 'tm'] + [col for col in df.columns if 'datetime' in str(df[col].dtype)]
    X_train = train_df.drop(columns=cols_to_drop, errors='ignore')
    y_train = train_df[target_column]
    X_val = val_df.drop(columns=cols_to_drop, errors='ignore')
    y_val = val_df[target_column]
    X_test = test_df.drop(columns=cols_to_drop, errors='ignore')
    y_test = test_df[target_column]

    numerical_cols = X_train.select_dtypes(include=np.number).columns.tolist()
    scaler = StandardScaler()
    if numerical_cols:
        # train 데이터에 맞춰 스케일링하고, val/test 데이터는 그것으로 변환만 수행
        X_train[numerical_cols] = scaler.fit_transform(X_train[numerical_cols])
        X_val[numerical_cols] = scaler.transform(X_val[numerical_cols])
        X_test[numerical_cols] = scaler.transform(X_test[numerical_cols])
        print(f"스케일링 적용 피처: {len(numerical_cols)}개")

    print("데이터 분할 및 스케일링 완료:")
    print(f"  - Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
    print(f"  - 전체 피처 수: {X_train.shape[1]}")
    return X_train, X_val, X_test, y_train, y_val, y_test, scaler


def save_processed_data(output_dir, X_train, X_val, X_test, y_train, y_val, y_test, scaler):
    """전처리된 데이터와 스케일러를 파일로 저장"""
    print(f"\n--- 전처리된 데이터 저장 시작 ---")
    os.makedirs(output_dir, exist_ok=True)
    print(f"저장 경로: '{output_dir}'")

    try:
        # 데이터프레임으로 다시 합쳐서 저장 (X와 y를 합쳐야 확인하기 용이)
        train_data = pd.concat([X_train, y_train], axis=1)
        val_data = pd.concat([X_val, y_val], axis=1)
        test_data = pd.concat([X_test, y_test], axis=1)

        # CSV 파일로 저장 (utf-8-sig는 Excel에서 한글 깨짐 방지)
        train_data.to_csv(os.path.join(output_dir, 'train_data.csv'), index=False, encoding='utf-8-sig')
        val_data.to_csv(os.path.join(output_dir, 'validation_data.csv'), index=False, encoding='utf-8-sig')
        test_data.to_csv(os.path.join(output_dir, 'test_data.csv'), index=False, encoding='utf-8-sig')
        print("✅ Train/Validation/Test 데이터셋 CSV 저장 완료.")

        # 스케일러 객체 저장
        joblib.dump(scaler, os.path.join(output_dir, 'scaler.joblib'))
        print("✅ StandardScaler 객체 저장 완료.")

    except Exception as e:
        print(f"❌ 데이터 저장 중 오류 발생: {e}")

    print("--- 데이터 저장 완료 ---")


def load_weather_alert_data(alert_path):
    """기상 특보 데이터 로드 함수"""
    encodings_to_try = ['utf-8', 'utf-8-sig', 'cp949', 'euc-kr']

    for encoding in encodings_to_try:
        try:
            df = pd.read_csv(alert_path, encoding=encoding, index_col=False)
            print(f"✅ '{os.path.basename(alert_path)}' 파일을 '{encoding}' 인코딩으로 읽기 성공.")
            return df
        except (UnicodeDecodeError, pd.errors.ParserError):
            continue

    print(f"❌ '{os.path.basename(alert_path)}' 파일을 다음 인코딩으로 읽을 수 없습니다: {encodings_to_try}")
    return pd.DataFrame()


# ==============================================================================
# ===== 메인 실행 블록: 스크립트를 실행하면 아래 코드가 동작합니다 =====
# ==============================================================================
if __name__ == '__main__':

    # --- 설정 ---
    BASE_DIR = os.path.dirname(__file__)
    # 경로를 실제 프로젝트 구조에 맞게 조정합니다.
    # src 폴더 안에 preprocess.py가 있다고 가정합니다.
    DATA_DIR = os.path.join(BASE_DIR, '..', 'data')

    CALL119_TRAIN_PATH = os.path.join(DATA_DIR, 'call119_train1.csv')
    CAT119_TRAIN_PATH = os.path.join(DATA_DIR, 'cat119_train1.csv')
    ALERT_TRAIN_PATH = os.path.join(DATA_DIR, 'weather_alert.csv')
    PEOPLE_RAW_PATH = os.path.join(DATA_DIR, '2023_people.csv')
    OUTPUT_DIRECTORY = './preprocessed_data'

    TRAFFIC_FILES_INFO = [
        {'path': os.path.join(DATA_DIR, 'Report_2020.csv'), 'year': 2020},
        {'path': os.path.join(DATA_DIR, 'Report_2021.csv'), 'year': 2021},
        {'path': os.path.join(DATA_DIR, 'Report_2022.csv'), 'year': 2022},
        {'path': os.path.join(DATA_DIR, 'Report_2023.csv'), 'year': 2023},
    ]

    # === 실행 ===
    # 1. 데이터 로드
    print("--- 데이터 로드 시작 ---")
    call_df, cat_df = load_data(CALL119_TRAIN_PATH, CAT119_TRAIN_PATH)

    # 기상 특보 데이터 로드
    alert_df = load_weather_alert_data(ALERT_TRAIN_PATH)
    people_df = preprocess_people_data(PEOPLE_RAW_PATH)

    all_traffic_dfs = []
    for file_info in TRAFFIC_FILES_INFO:
        if os.path.exists(file_info['path']):
            df_year = preprocess_traffic_data(file_info['path'], file_info['year'])
            all_traffic_dfs.append(df_year)
        else:
            print(f"⚠️ 경고: '{file_info['path']}' 파일을 찾을 수 없어 건너뜁니다.")

    if all_traffic_dfs:
        traffic_df = pd.concat(all_traffic_dfs, ignore_index=True)
        print(f"\n✅ 모든 연도 교통사고 데이터 통합 완료. 전체 크기: {traffic_df.shape}")
    else:
        traffic_df = pd.DataFrame()  # 빈 데이터프레임으로 초기화
        print("\n⚠️ 처리할 교통사고 데이터가 없습니다.")

    # 2. 데이터 전처리
    print("\n--- 데이터 전처리 시작 ---")
    preprocessed_df = preprocess_data(call_df, cat_df, alert_df, traffic_df, people_df)

    # 3. 피처 인코딩
    print("\n--- 피처 인코딩 시작 ---")
    encoded_df = encode_features(preprocessed_df)

    # 4. 사람이 읽기 쉬운 형태의 데이터 저장
    print(f"\n--- 사람이 읽기 쉬운 형태의 데이터 저장 시작 ---")
    try:
        os.makedirs(OUTPUT_DIRECTORY, exist_ok=True)
        human_readable_path = os.path.join(OUTPUT_DIRECTORY, 'preprocessed_human_readable_traffic.csv')
        encoded_df.to_csv(human_readable_path, index=False, encoding='utf-8-sig')
        print(f"✅ 스케일링 전 데이터 저장 완료: '{human_readable_path}'")
    except Exception as e:
        print(f"❌ 읽기 쉬운 데이터 저장 중 오류 발생: {e}")

    # 5. 데이터 분할 및 스케일링
    print("\n--- 데이터 분할 및 스케일링 시작 ---")
    X_train, X_val, X_test, y_train, y_val, y_test, scaler_obj = split_data_time_series(
        encoded_df,
        target_column='call_count'
    )

    # 6. 최종 결과물(모델 학습용) 파일로 저장
    save_processed_data(
        OUTPUT_DIRECTORY,
        X_train, X_val, X_test,
        y_train, y_val, y_test,
        scaler_obj
    )

    print("\n🎉 전체 전처리 과정이 완료되었습니다!")
    print(f"결과 파일들이 '{OUTPUT_DIRECTORY}' 폴더에 저장되었습니다.")