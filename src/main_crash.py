# main.py - 119 신고 건수 예측 모델링 프로젝트 (모든 기능 및 상세 로그 유지 최종본)

import pandas as pd
import os
import sys
import joblib
import argparse
from datetime import datetime

# 경로 설정을 위해 src 폴더를 path에 추가
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

# 필요한 모든 함수 임포트
from src.pre_esm_crash import (load_data, preprocess_data, encode_features,
                         split_data_time_series, preprocess_people_data,
                         preprocess_traffic_data)
from src.train_ml_model_crash import run_stacking_model


class Config:
    """프로젝트의 모든 설정을 관리하는 클래스"""
    BASE_DIR = os.path.dirname(__file__)
    DATA_DIR = os.path.join(BASE_DIR, '..', 'data')
    RESULTS_DIR = os.path.join(BASE_DIR, 'results')
    MODELS_DIR = os.path.join(BASE_DIR, 'models')

    # 입력 데이터 파일 경로
    CALL119_TRAIN_PATH = os.path.join(DATA_DIR, 'call119_train1.csv')
    CAT119_TRAIN_PATH = os.path.join(DATA_DIR, 'cat119_train1.csv')
    ALERT_TRAIN_PATH = os.path.join(DATA_DIR, 'weather_alert.csv')
    PEOPLE_TRAIN_PATH = os.path.join(DATA_DIR, '2023_people.csv')

    TRAFFIC_FILES_INFO = [
        {'path': os.path.join(DATA_DIR, 'Report_2020.csv'), 'year': 2020},
        {'path': os.path.join(DATA_DIR, 'Report_2021.csv'), 'year': 2021},
        {'path': os.path.join(DATA_DIR, 'Report_2022.csv'), 'year': 2022},
        {'path': os.path.join(DATA_DIR, 'Report_2023.csv'), 'year': 2023},
    ]

    # 모델 및 아티팩트 저장 경로
    STACKING_MODEL_PATH = os.path.join(MODELS_DIR, 'optimized_stacking_regressor_crash.pkl')
    MODEL_COLUMNS_PATH = os.path.join(MODELS_DIR, 'ml_model_columns_crash.pkl')
    SCALER_PATH = os.path.join(MODELS_DIR, 'ml_scaler_crash.pkl')

    # 모델 학습 파라미터
    TARGET_COLUMN = 'call_count'
    TRAIN_RATIO = 0.7
    VAL_RATIO = 0.15
    RANDOM_STATE = 42


def setup_directories():
    """결과 및 모델 저장을 위한 디렉토리 생성"""
    for directory in [Config.RESULTS_DIR, Config.MODELS_DIR]:
        os.makedirs(directory, exist_ok=True)


def check_data_files():
    """데이터 파일 존재 여부 확인"""
    required_files = [
        (Config.CALL119_TRAIN_PATH, 'call119 훈련 데이터'),
        (Config.CAT119_TRAIN_PATH, 'cat119 훈련 데이터'),
    ]
    for file_info in Config.TRAFFIC_FILES_INFO:
        if os.path.exists(file_info['path']):
            required_files.append((file_info['path'], f"{file_info['year']}년 교통사고 데이터"))

    all_files_exist = True
    for file_path, description in required_files:
        if not os.path.exists(file_path):
            print(f"   - ❌ '{description}' 파일을 찾을 수 없습니다: {file_path}")
            all_files_exist = False
        else:
            print(f"   - ✅ '{description}' 파일 확인 완료.")
    return all_files_exist


def print_project_info():
    """프로젝트 정보 출력"""
    print("=" * 80)
    print("🚨 119 신고 건수 예측 모델링 프로젝트 (모든 기능 및 상세 로그 유지) 🚨")
    print("=" * 80)


def load_and_validate_data():
    """모든 데이터 소스를 로드하고 검증"""
    call_df, cat_df = load_data(Config.CALL119_TRAIN_PATH, Config.CAT119_TRAIN_PATH)

    alert_df = pd.read_csv(Config.ALERT_TRAIN_PATH, encoding='utf-8') if os.path.exists(
        Config.ALERT_TRAIN_PATH) else pd.DataFrame()
    print(f"   - 기상특보 데이터 로드: {alert_df.shape}")

    people_df = preprocess_people_data(Config.PEOPLE_TRAIN_PATH) if os.path.exists(
        Config.PEOPLE_TRAIN_PATH) else pd.DataFrame()

    traffic_dfs = [preprocess_traffic_data(f['path'], f['year']) for f in Config.TRAFFIC_FILES_INFO if
                   os.path.exists(f['path'])]
    traffic_df = pd.concat(traffic_dfs, ignore_index=True) if traffic_dfs else pd.DataFrame()
    print(f"   - 통합 교통사고 데이터 생성: {traffic_df.shape}")

    return call_df, cat_df, alert_df, traffic_df, people_df


def preprocess_and_encode_data(call119_df, cat119_df, alert_df, traffic_df, people_df):
    """데이터 통합 전처리 및 인코딩"""
    processed_df = preprocess_data(call119_df, cat119_df, alert_df, traffic_df, people_df)
    final_df = encode_features(processed_df)
    return final_df


def split_and_scale_data(final_df):
    """데이터 분할 및 스케일링"""
    return split_data_time_series(
        final_df, Config.TARGET_COLUMN, Config.TRAIN_RATIO, Config.VAL_RATIO
    )


def save_preprocessing_artifacts(X_train, scaler):
    """전처리 관련 아티팩트 저장"""
    joblib.dump(X_train.columns, Config.MODEL_COLUMNS_PATH)
    joblib.dump(scaler, Config.SCALER_PATH)


def train_model(X_train, X_val, X_test, y_train, y_val, y_test):
    """모델 학습"""
    return run_stacking_model(
        X_train, X_val, X_test, y_train, y_val, y_test,
        results_dir=Config.RESULTS_DIR,
        models_dir=Config.MODELS_DIR
    )


def print_completion_summary(results):
    """프로젝트 완료 요약 출력"""
    print("\n" + "=" * 80)
    print("🎉 모델링 프로젝트 완료!")
    print("=" * 80)
    if results and 'metrics' in results:
        metrics = results['metrics']
        print(f"📊 최종 성능 (테스트 세트):")
        print(f"   - MAE: {metrics['MAE']:.4f}, RMSE: {metrics['RMSE']:.4f}, R²: {metrics['R2']:.4f}")

    print("\n📁 생성된 주요 파일:")
    print(f"   - 모델: {Config.STACKING_MODEL_PATH}")
    print(f"   - 결과분석 그래프 및 요약: '{Config.RESULTS_DIR}' 폴더")

    print(f"\n🚀 다음 단계: 'predict_esm.py'를 실행하여 예측을 수행하세요.")
    print("=" * 80)


def main(args=None):
    """메인 실행 함수"""
    try:
        print_project_info()

        print("1. 디렉토리 설정 확인...")
        setup_directories()
        print("✅ 디렉토리 준비 완료.")

        print("\n2. 데이터 파일 존재 여부 확인...")
        if not check_data_files():
            print("\n❌ 필수 데이터 파일이 없어 프로그램을 종료합니다.")
            return False

        print("\n3. 모든 데이터 소스 로드 및 개별 전처리 시작")
        print("-" * 50)
        datasets = load_and_validate_data()
        call_df, cat_df, alert_df, traffic_df, people_df = datasets
        print("✅ 모든 데이터 로드 완료.")

        print("\n4. 데이터 통합 전처리 및 인코딩 시작")
        print("-" * 50)
        final_df = preprocess_and_encode_data(*datasets)
        print("✅ 전처리 및 인코딩 완료.")

        print("\n5. 데이터 분할 및 스케일링 시작")
        print("-" * 50)
        X_train, X_val, X_test, y_train, y_val, y_test, scaler = split_and_scale_data(final_df)
        print("✅ 데이터 분할 및 스케일링 완료.")

        print("\n6. 전처리 아티팩트(컬럼, 스케일러) 저장 시작")
        print("-" * 50)
        save_preprocessing_artifacts(X_train, scaler)
        print("✅ 아티팩트 저장 완료.")

        print("\n7. 모델 학습 시작")
        print("-" * 50)
        results = train_model(X_train, X_val, X_test, y_train, y_val, y_test)
        print("✅ 모델 학습 완료.")

        print_completion_summary(results)
        return True

    except Exception as e:
        print(f"\n❌ 예상치 못한 오류가 발생했습니다: {e}")
        import traceback
        traceback.print_exc()
        return False


def parse_arguments():
    """명령행 인수 파싱"""
    parser = argparse.ArgumentParser(description='119 신고 건수 예측 모델 학습')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_arguments()
    success = main(args)
    sys.exit(0 if success else 1)