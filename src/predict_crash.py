import pandas as pd
import numpy as np
import joblib
import os
import sys

# 경로 설정을 위해 src 폴더를 path에 추가
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

from src.pre_esm_crash import load_data, preprocess_data, encode_features, preprocess_people_data, preprocess_traffic_data

RESULTS_DIR = 'predictions'
MODEL_DIR = 'models'

BASE_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(BASE_DIR, '..', 'data')
TEST_CALL_PATH = os.path.join(DATA_DIR, 'call119_test.csv')
TEST_CAT_PATH = os.path.join(DATA_DIR, 'cat119_test.csv')
TEST_ALERT_PATH = os.path.join(DATA_DIR, '2024_weather_alert.csv')
TEST_PEOPLE_PATH = os.path.join(DATA_DIR, '2023_people.csv')


TEST_TRAFFIC_FILES_INFO = [
    {'path': os.path.join(DATA_DIR, 'Report_2023.csv'), 'year': 2023},
    {'path': os.path.join(DATA_DIR, 'Report_2024.csv'), 'year': 2024},
]
# 불러올 모델 파일 경로를 스태킹 모델로 변경합니다.
MODEL_PATH = os.path.join(MODEL_DIR, 'optimized_stacking_regressor_alert.pkl')
MODEL_COLUMNS_PATH = os.path.join(MODEL_DIR, 'ml_model_columns_crash.pkl')
SCALER_PATH = os.path.join(MODEL_DIR, 'ml_scaler_crash.pkl')


def predict():
    """
    학습된 Stacking 앙상블 모델을 사용하여 예측을 수행하는 메인 함수
    """
    print("--- 앙상블 모델 예측 절차 시작 ---")
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # 1. 모델 및 전처리기(컬럼, 스케일러) 로드
    print("\n1. 학습된 Stacking 모델, 컬럼, 스케일러 로드...")
    try:
        model = joblib.load(MODEL_PATH)
        model_columns = joblib.load(MODEL_COLUMNS_PATH)
        scaler = joblib.load(SCALER_PATH)
        print(f"✅ 모델 로드 완료: {os.path.basename(MODEL_PATH)}")
        print(f"   - 예상 피처 수: {len(model_columns)}")
    except FileNotFoundError as e:
        print(f"❌ 오류: 모델 관련 파일을 찾을 수 없습니다. 'main_esm.py'를 먼저 실행하여 모델을 학습시키세요.")
        print(f"   - 누락된 파일: {e.filename}")
        return

    # 2. 테스트 데이터 로드
    print("\n2. 테스트 데이터 로드...")
    try:
        call119_test_df, cat119_test_df = load_data(TEST_CALL_PATH, TEST_CAT_PATH)

        # 기상 특보 데이터 로드
        alert_test_df = pd.DataFrame()
        if os.path.exists(TEST_ALERT_PATH):
            try:

                for encoding in ['utf-8', 'utf-8-sig', 'cp949', 'euc-kr']:
                    try:
                        alert_test_df = pd.read_csv(TEST_ALERT_PATH, encoding=encoding, index_col=False)
                        print(f"✅ 테스트용 기상특보 데이터 로드: {os.path.basename(TEST_ALERT_PATH)}")
                        break
                    except (UnicodeDecodeError, pd.errors.ParserError):
                        continue
                if alert_test_df.empty:
                    print(f"❌ '{os.path.basename(TEST_ALERT_PATH)}' 파일을 읽을 수 없습니다.")
            except Exception as e:
                print(f"❌ 테스트용 기상특보 데이터 로드 중 오류 발생: {e}")
                alert_test_df = pd.DataFrame()
        else:
            print("⚠️ 테스트용 기상특보 파일이 없어, 특보 없음을 가정하고 진행합니다.")
            alert_test_df = pd.DataFrame()

        # 인구 데이터 로드 및 전처리
        people_test_df = pd.DataFrame()
        if os.path.exists(TEST_PEOPLE_PATH):
            try:
                people_test_df = preprocess_people_data(TEST_PEOPLE_PATH)
                print(f"✅ 테스트용 인구 데이터 로드 및 전처리: {os.path.basename(TEST_PEOPLE_PATH)}")
            except Exception as e:
                print(f"❌ 테스트용 인구 데이터 전처리 중 오류 발생: {e}. 인구 데이터 없이 진행합니다.")
                people_test_df = pd.DataFrame()
        else:
            print(f"⚠️ 테스트용 인구 데이터 파일이 존재하지 않습니다: {TEST_PEOPLE_PATH}. 인구 데이터 없이 진행합니다.")
        traffic_dfs = []
        for file_info in TEST_TRAFFIC_FILES_INFO:
            if os.path.exists(file_info['path']):
                try:
                    df_year = preprocess_traffic_data(file_info['path'], file_info['year'])
                    traffic_dfs.append(df_year)
                except Exception as e:
                    print(f"❌ {file_info['year']}년 교통사고 데이터 처리 중 오류: {e}")
            else:
                print(f"⚠️  {file_info['year']}년 교통사고 데이터 파일을 찾을 수 없습니다: {file_info['path']}")

        if traffic_dfs:
            traffic_test_df = pd.concat(traffic_dfs, ignore_index=True)
            print(f"✅ 테스트용 교통사고 데이터 통합 완료: {traffic_test_df.shape}")
        else:
            traffic_test_df = pd.DataFrame()
            print("⚠️ 처리할 교통사고 데이터가 없어, 해당 피처 없이 예측을 진행합니다.")

    except FileNotFoundError as e:
        print(f"❌ 오류: 테스트 데이터 파일을 찾을 수 없습니다.")
        print(f"   - 누락된 파일: {e.filename}")
        return
    except Exception as e:
        print(f"❌ 데이터 로드 중 오류 발생: {e}")
        return


    print("\n3. 데이터 전처리 및 인코딩...")
    try:
        # 전처리: weather_alert와 people 데이터 전달
        processed_test_df = preprocess_data(
            call119_test_df.copy(),
            cat119_test_df.copy(),
            alert_test_df,
            traffic_test_df,
            people_test_df
        )
        print(f"✅ 전처리 완료: {processed_test_df.shape}")

        # 인코딩: 학습 시 사용했던 함수와 동일한 함수를 사용합니다.
        final_test_df = encode_features(processed_test_df)
        print(f"✅ 인코딩 완료: {final_test_df.shape}")

    except Exception as e:
        print(f"❌ 전처리 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        return

        # ID 컬럼 처리 (제출용)
    submission_df = pd.DataFrame()
    # 'id' 컬럼이 원본 데이터에 있는지 확인 (clean_column_names로 소문자화됨)
    if 'id' in final_test_df.columns:
        submission_df['ID'] = final_test_df['id']
    elif 'ID' in final_test_df.columns:  # 원본이 대문자일 경우도 대비
        submission_df['ID'] = final_test_df['ID']
    else:
        # ID 컬럼이 없는 경우, 순차적인 임시 ID 생성
        submission_df['ID'] = ["TEST_" + str(i) for i in range(len(final_test_df))]
        print(f"⚠️ ID 컬럼이 없어 임시 ID를 생성합니다. (예: TEST_0, TEST_1...)")

    # 4. 피처 정렬 및 스케일링
    print("\n4. 피처 정렬 및 스케일링 적용...")
    # 4-1. 훈련 데이터의 컬럼 순서와 동일하게 맞추고, 없는 컬럼은 0으로 채웁니다.
    X_test = final_test_df.reindex(columns=model_columns, fill_value=0)
    print(f"✅ 피처 정렬 완료. 최종 피처 수: {X_test.shape[1]}")

    # 4-2. 저장된 스케일러로 데이터를 변환(transform)합니다.
    try:

        numerical_cols_from_scaler = scaler.feature_names_in_
        cols_to_scale = [col for col in numerical_cols_from_scaler if col in X_test.columns]

        if cols_to_scale:

            X_test[cols_to_scale] = scaler.transform(X_test[cols_to_scale])
            print(f"✅ 스케일링 적용 완료: {len(cols_to_scale)}개 컬럼")
        else:
            print("⚠️ 스케일링할 수치형 컬럼이 없거나, 스케일러가 학습된 컬럼과 일치하지 않습니다.")
    except Exception as e:
        print(f"❌ 스케일링 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        return

        # 5. 예측 수행
    print("\n5. 예측 수행...")
    try:
        predictions = model.predict(X_test)
        print(f"✅ 예측 완료: {len(predictions)} 건")
    except Exception as e:
        print(f"❌ 예측 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        return

    # 6. 결과 후처리 및 저장
    print("\n6. 결과 저장...")
    # 예측값은 0 이상이어야 하고, 정수 형태여야 합니다.
    predictions_cleaned = np.maximum(0, predictions).round().astype(int)
    submission_df['prediction'] = predictions_cleaned

    output_path = os.path.join(RESULTS_DIR, 'submission_pe10_crash.csv')
    submission_df.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"✅ 최종 예측 결과가 '{output_path}'에 저장되었습니다.")

    print(f"\n📊 예측 결과 요약:")
    print(f"   - 총 예측 건수: {len(predictions_cleaned)}")
    print(f"   - 예측값 범위: {predictions_cleaned.min()} ~ {predictions_cleaned.max()}")
    print(f"   - 예측값 평균: {predictions_cleaned.mean():.2f}")


if __name__ == '__main__':
    predict()
