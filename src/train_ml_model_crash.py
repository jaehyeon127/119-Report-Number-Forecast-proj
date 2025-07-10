import time
import pandas as pd
import xgboost
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import time
import joblib
from collections import Counter
from lightgbm import LGBMRegressor
from sklearn.linear_model import HuberRegressor, RidgeCV
from sklearn.ensemble import StackingRegressor
import warnings
import shap
import optuna

def evaluate_model(model, X, y, model_name, data_type="Test"):
    """
    모델을 평가하고 결과를 출력합니다.
    """
    start_time = time.time()
    y_pred = model.predict(X)
    end_time = time.time()

    mae = mean_absolute_error(y, y_pred)
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    r2 = r2_score(y, y_pred)

    # 추가 평가 지표
    mape = np.mean(np.abs((y - y_pred) / np.maximum(y, 1))) * 100  # 0으로 나누기 방지

    print(f"\n[{model_name} - {data_type} 평가]")
    print(f"  - 예측 시간: {end_time - start_time:.4f} 초")
    print(f"  - MAE: {mae:.4f}")
    print(f"  - RMSE: {rmse:.4f}")
    print(f"  - R2: {r2:.4f}")
    print(f"  - MAPE: {mape:.2f}%")

    # 예측값 분포 분석
    print(f"  - 실제값 범위: {y.min():.0f} ~ {y.max():.0f}")
    print(f"  - 예측값 범위: {y_pred.min():.2f} ~ {y_pred.max():.2f}")
    print(f"  - 음수 예측값: {(y_pred < 0).sum()}개")

    return {'MAE': mae, 'RMSE': rmse, 'R2': r2, 'MAPE': mape, 'y_pred': y_pred}


def plot_predictions(y_true, y_pred, model_name,results_dir):
    """
    실제값 vs 예측값, 잔차 분포를 시각화합니다.
    """
    plt.figure(figsize=(16, 8))

    # 한글 폰트 설정 (선택사항)
    try:
        plt.rc('font', family='Malgun Gothic')
    except:
        print("경고: 'Malgun Gothic' 폰트를 찾을 수 없습니다. 그래프의 한글이 깨질 수 있습니다.")

    plt.suptitle(f'{model_name} - 예측 성능 분석', fontsize=16)

    # 1. 실제값 vs 예측값 산점도
    plt.subplot(2, 2, 1)
    plt.scatter(y_true, y_pred, alpha=0.6, s=10)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], '--', color='red', linewidth=2)
    plt.title('실제값 vs. 예측값')
    plt.xlabel('실제값')
    plt.ylabel('예측값')
    plt.grid(True, alpha=0.3)

    # R² 값 표시
    r2 = r2_score(y_true, y_pred)
    plt.text(0.05, 0.95, f'R² = {r2:.3f}', transform=plt.gca().transAxes,
             bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

    # 2. 잔차 분포
    plt.subplot(2, 2, 2)
    residuals = y_true - y_pred
    sns.histplot(residuals, kde=True, bins=30)
    plt.title('잔차 분포')
    plt.xlabel('잔차 (실제값 - 예측값)')
    plt.ylabel('빈도')
    plt.grid(True, alpha=0.3)

    # 잔차 통계 표시
    plt.axvline(residuals.mean(), color='red', linestyle='--', alpha=0.7, label=f'평균: {residuals.mean():.2f}')
    plt.legend()

    # 3. 예측값 분포
    plt.subplot(2, 2, 3)
    plt.hist(y_true, bins=30, alpha=0.7, label='실제값', color='blue')
    plt.hist(y_pred, bins=30, alpha=0.7, label='예측값', color='orange')
    plt.title('실제값 vs 예측값 분포')
    plt.xlabel('값')
    plt.ylabel('빈도')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 4. 잔차 vs 예측값 (이상치 탐지)
    plt.subplot(2, 2, 4)
    plt.scatter(y_pred, residuals, alpha=0.6, s=10)
    plt.axhline(y=0, color='red', linestyle='--', alpha=0.7)
    plt.title('잔차 vs 예측값')
    plt.xlabel('예측값')
    plt.ylabel('잔차')
    plt.grid(True, alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    os.makedirs(results_dir, exist_ok=True)  # 2. 하드코딩된 'results' 대신 인자로 받은 results_dir를 사용합니다.
    save_path = os.path.join(results_dir, f'{model_name}_prediction_analysis.png')  # 3. 저장 경로를 조합합니다.
    plt.savefig(save_path, dpi=300, bbox_inches='tight')  # 4. 조합된 경로에 저장합니다.
    plt.close()

    print(f"✅ 예측 분석 그래프 저장: results/{model_name}_prediction_analysis.png")


def plot_feature_importance(model, feature_names, model_name,results_dir):
    """
    피처 중요도를 시각화합니다.
    """
    importance = model.feature_importances_
    df_importance = pd.DataFrame({'feature': feature_names, 'importance': importance})
    df_importance = df_importance.sort_values('importance', ascending=False)

    # 상위 30개 피처
    top_features = df_importance.head(30)

    plt.figure(figsize=(12, 10))
    try:
        plt.rc('font', family='Malgun Gothic')
        plt.rcParams['axes.unicode_minus'] = False
    except:
        print("경고: 'Malgun Gothic' 폰트를 찾을 수 없습니다. 그래프의 한글이 깨질 수 있습니다.")

    sns.barplot(x='importance', y='feature', data=top_features, palette='viridis')
    plt.title(f'{model_name} - 상위 30개 피처 중요도')
    plt.xlabel('중요도')
    plt.ylabel('피처')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    os.makedirs(results_dir, exist_ok=True)
    save_path = os.path.join(results_dir, f'{model_name}_feature_importance.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ 피처 중요도 그래프 저장: results/{model_name}_feature_importance.png")

    # 피처 중요도 상위 20개 출력
    print(f"\n🔍 상위 20개 중요 피처:")
    for i, (_, row) in enumerate(top_features.head(20).iterrows(), 1):
        print(f"  {i:2d}. {row['feature']:<40} : {row['importance']:.4f}")

        # 피처 유형별 중요도 분석
    print(f"\n📊 피처 유형별 중요도 분석:")
    feature_types = {
        'accident_': '교통사고',
        'cat_': '신고유형',
        'subcat_': '세부유형',
        'address_': '지역',
        'population_': '인구',
        'ta_': '기온',
        'hm_': '습도',
        'ws_': '풍속',
        'rn_': '강수',
        'alert_': '기상특보',
        'year': '시간_년',
        'month': '시간_월',
        'day': '시간_일',
        'dayofweek': '시간_요일',
        'quarter': '시간_분기',
        'is_weekend': '시간_주말여부',
        'season_': '시간_계절',
        'is_holiday': '시간_공휴일여부'
    }

    # 각 피처가 어떤 유형에 속하는지 매핑
    feature_type_map = {}
    for col in feature_names:
        found = False
        for prefix, type_name in feature_types.items():
            if col.startswith(prefix):
                feature_type_map[col] = type_name
                found = True
                break
        if not found:
            feature_type_map[col] = '기타'

    # 유형별 중요도 합산
    type_importance_sum = Counter()
    for _, row in df_importance.iterrows():
        feature_name = row['feature']
        importance_value = row['importance']
        type_name = feature_type_map.get(feature_name, '기타')
        type_importance_sum[type_name] += importance_value

    for type_name, total_importance in type_importance_sum.most_common():
        count = sum(1 for f in feature_names if feature_type_map.get(f) == type_name)
        avg_importance = total_importance / count if count > 0 else 0
        print(f"  - {type_name:<8}: 총 {total_importance:.4f} (평균: {avg_importance:.4f}, 개수: {count})")


def analyze_data_quality(X_train, y_train, X_val, y_val, X_test, y_test):
    """
    데이터 품질을 분석합니다.
    """
    print(f"\n📊 데이터 품질 분석:")
    print(f"  - 훈련 데이터: {X_train.shape[0]:,}개 샘플, {X_train.shape[1]:,}개 피처")
    print(f"  - 검증 데이터: {X_val.shape[0]:,}개 샘플")
    print(f"  - 테스트 데이터: {X_test.shape[0]:,}개 샘플")

    # 타겟 변수 분포
    print(f"\n📈 타겟 변수(call_count) 분포:")
    for name, y in [('훈련', y_train), ('검증', y_val), ('테스트', y_test)]:
        print(f"  - {name}: 평균={y.mean():.2f}, 중간값={y.median():.2f}, "
              f"최소={y.min()}, 최대={y.max()}, 표준편차={y.std():.2f}")

    # 피처 유형 분석
    print(f"\n🔍 피처 유형 분석:")
    feature_counts = Counter()
    for col in X_train.columns:
        if col.startswith('cat_'):
            feature_counts['신고유형'] += 1
        elif col.startswith('subcat_'):
            feature_counts['세부유형'] += 1
        elif col.startswith('address_'):
            feature_counts['지역'] += 1
        elif any(col.startswith(x) for x in ['ta_', 'hm_', 'ws_', 'rn_']):
            feature_counts['기상'] += 1
        elif any(col.startswith(x) for x in ['year_', 'month_', 'day_', 'dayofweek_', 'quarter_', 'season_']):
            feature_counts['시간'] += 1
        else:
            feature_counts['기타'] += 1

    for feature_type, count in feature_counts.items():
        print(f"  - {feature_type}: {count}개")


def analyze_shap_values(model, X_data, feature_names,results_dir):
    """
    SHAP를 사용하여 모델의 예측을 해석하고 주요 그래프를 저장합니다.
    """
    print("\n[+ 추가 단계] SHAP 분석 시작...")
    try:

        explainer = shap.TreeExplainer(model)

        sample_size = min(5000, X_data.shape[0])
        sampled_X_data = shap.utils.sample(X_data, sample_size, random_state=42)  # random_state 추가
        print(f"  - SHAP 값 계산 중 (샘플 크기: {sample_size})...")
        shap_values = explainer(sampled_X_data)
        print("  - SHAP 값 계산 완료.")

        os.makedirs(results_dir, exist_ok=True)

        # 1. SHAP 요약 그래프 (Summary Plot) 저장
        print("  - SHAP 요약 그래프 생성 중...")
        plt.figure(figsize=(10, 8))  # 그래프 크기 조정
        shap.summary_plot(shap_values, sampled_X_data, show=False, feature_names=feature_names)
        plt.tight_layout()
        summary_plot_path = os.path.join(results_dir, 'shap_summary_plot.png')  # results_dir 사용
        plt.savefig(summary_plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ SHAP 요약 그래프 저장: {summary_plot_path}")

        # 2. SHAP 의존성 그래프 (Dependence Plot) 저장
        print("  - SHAP 의존성 그래프 생성 중 (상위 5개 피처)...")
        vals = np.abs(shap_values.values).mean(0)
        feature_importance = pd.DataFrame(list(zip(feature_names, vals)), columns=['feature', 'importance'])
        feature_importance.sort_values(by=['importance'], ascending=False, inplace=True)
        top_features = feature_importance['feature'].head(5).tolist()

        for feature in top_features:
            plt.figure(figsize=(8, 6))  # 그래프 크기 조정
            # sampled_X_data를 사용하여 dependence_plot 생성
            shap.dependence_plot(feature, shap_values.values, sampled_X_data, feature_names=feature_names,
                                 interaction_index="auto", show=False)
            plt.tight_layout()
            dependence_plot_path = os.path.join(results_dir, f'shap_dependence_plot_{feature}.png')  # results_dir 사용
            plt.savefig(dependence_plot_path, dpi=300, bbox_inches='tight')
            plt.close()
        print(f"✅ SHAP 의존성 그래프 저장 완료.")

    except ImportError:
        print("❌ SHAP 라이브러리가 설치되어 있지 않습니다. pip install shap 명령으로 설치하세요.")
    except Exception as e:
        print(f"❌ SHAP 분석 중 오류 발생: {e}")
    import traceback

    traceback.print_exc()


def run_stacking_model(X_train, X_val, X_test, y_train, y_val, y_test,results_dir, models_dir):
    """
    Optuna를 사용하여 Stacking 모델의 하이퍼파라미터를 최적화하고,
    최적의 파라미터로 최종 모델을 학습 및 평가, 저장합니다.
    """
    print("\n" + "=" * 60)
    print("🚀 Optuna 기반 Stacking 앙상블 최적화 시작")
    print("=" * 60)

    # --- 1. Objective 함수 정의 ---
    def objective(trial):
        xgb_max_depth = trial.suggest_int('xgb_max_depth', 4, 8)
        xgb_learning_rate = trial.suggest_float('xgb_learning_rate', 0.01, 0.3, log=True)
        xgb_subsample = trial.suggest_float('xgb_subsample', 0.6, 1.0)
        xgb_colsample_bytree = trial.suggest_float('xgb_colsample_bytree', 0.6, 1.0)
        lgbm_max_depth = trial.suggest_int('lgbm_max_depth', 4, 8)
        lgbm_learning_rate = trial.suggest_float('lgbm_learning_rate', 0.01, 0.3, log=True)

        base_models = [
            ('xgboost', xgboost.XGBRegressor(random_state=42, n_estimators=1000, n_jobs=-1, max_depth=xgb_max_depth,
                                             learning_rate=xgb_learning_rate, subsample=xgb_subsample,
                                             colsample_bytree=xgb_colsample_bytree)),
            ('lightgbm',
             LGBMRegressor(random_state=42, n_estimators=1000, n_jobs=-1, verbose=-1, max_depth=lgbm_max_depth,
                           learning_rate=lgbm_learning_rate)),
            ('huber', HuberRegressor(epsilon=1.35, max_iter=5000))
        ]
        stacking_model_trial = StackingRegressor(estimators=base_models, final_estimator=RidgeCV(), cv=3, n_jobs=-1)
        stacking_model_trial.fit(X_train, y_train)
        preds = stacking_model_trial.predict(X_val)
        rmse = np.sqrt(mean_squared_error(y_val, preds))
        return rmse

    # --- 2. Optuna Study 실행 ---
    print("\n[1단계] Optuna Study 실행... (n_trials 만큼 반복)")
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=50)

    print("✅ 최적화 완료!")
    print(f"  - 최적의 검증 RMSE: {study.best_value:.4f}")
    print("  - 최적의 하이퍼파라미터:")
    for key, value in study.best_params.items():
        print(f"    - {key}: {value}")

    # --- 3. 최적의 파라미터로 최종 모델 학습 ---
    print("\n[2단계] 찾은 최적의 파라미터로 최종 모델 학습")
    best_params = study.best_params
    final_base_models = [
        ('xgboost',
         xgboost.XGBRegressor(random_state=42, n_estimators=1000, n_jobs=-1, max_depth=best_params['xgb_max_depth'],
                              learning_rate=best_params['xgb_learning_rate'], subsample=best_params['xgb_subsample'],
                              colsample_bytree=best_params['xgb_colsample_bytree'])),
        ('lightgbm', LGBMRegressor(random_state=42, n_estimators=1000, n_jobs=-1, verbose=-1,
                                   max_depth=best_params['lgbm_max_depth'],
                                   learning_rate=best_params['lgbm_learning_rate'])),
        ('huber', HuberRegressor(epsilon=1.35, max_iter=5000))
    ]
    final_meta_model = RidgeCV()
    final_stacking_model = StackingRegressor(estimators=final_base_models, final_estimator=final_meta_model, cv=5,
                                             n_jobs=-1)

    print("  - 최종 모델 학습 중... (시간이 소요됩니다)")

    start_time = time.time()
    final_stacking_model.fit(X_train, y_train)
    end_time = time.time()

    print(f"✅ 최종 모델 학습 완료! (소요 시간: {end_time - start_time:.2f}초)")

    # --- 4. 최종 모델 평가 ---
    print("\n[3단계] 최종 모델 성능 평가")
    train_metrics = evaluate_model(final_stacking_model, X_train, y_train, "Optimized Stacking", "Train")
    val_metrics = evaluate_model(final_stacking_model, X_val, y_val, "Optimized Stacking", "Validation")
    test_metrics = evaluate_model(final_stacking_model, X_test, y_test, "Optimized Stacking", "Test")

    # --- 5. 최종 결과 시각화 및 저장 ---
    print("\n[4단계] 최종 결과 시각화 및 파일 저장")
    plot_predictions(y_test, test_metrics['y_pred'], "Optimized_Stacking_Regressor", results_dir)
    xgb_in_stack = final_stacking_model.named_estimators_['xgboost']
    plot_feature_importance(xgb_in_stack, X_train.columns.tolist(), "Optimized_Stacking_XGBoost_Features", results_dir)

    try:
        print("\n[5단계] SHAP 분석으로 모델 해석")
        xgb_in_stack = final_stacking_model.named_estimators_['xgboost']

        analyze_shap_values(xgb_in_stack, X_test, X_test.columns, results_dir)
    except Exception as e:
        print(f"❌ SHAP 분석 호출 중 오류: {e}")

    print("\n[6단계] 최종 모델 및 결과 요약 저장")
    os.makedirs(models_dir, exist_ok=True)

    model_path = os.path.join(models_dir, 'optimized_stacking_regressor_alert.pkl')
    joblib.dump(final_stacking_model, model_path)
    print(f"✅ 최종 Stacking 모델 저장 완료: {model_path}")


    summary_path = os.path.join(results_dir, 'optimized_stacking_summary.txt')
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write("=== Optimized Stacking 앙상블 모델 학습 결과 요약 ===\n\n")
        f.write(f"Optuna Best Validation RMSE: {study.best_value:.4f}\n")
        f.write(f"학습 시간: {end_time - start_time:.2f}초\n")
        f.write("Best Hyperparameters:\n")
        for key, value in best_params.items():
            f.write(f"  - {key}: {value}\n")
        f.write("\n")

        meta_coefs = final_stacking_model.final_estimator_.coef_
        f.write("Meta Model (RidgeCV) Coefficients:\n")
        for i, model_info in enumerate(final_base_models):
            f.write(f"  - {model_info[0]:<10}: {meta_coefs[i]:.4f}\n")
        f.write("\n")

        for phase, metrics in [('Train', train_metrics), ('Validation', val_metrics), ('Test', test_metrics)]:
            f.write(f"--- {phase} Set Performance ---\n")
            f.write(f"  MAE : {metrics['MAE']:.4f}\n")
            f.write(f"  RMSE: {metrics['RMSE']:.4f}\n")
            f.write(f"  R²  : {metrics['R2']:.4f}\n")
            f.write(f"  MAPE: {metrics['MAPE']:.2f}%\n\n")

    print(f"✅ 모델 요약 저장 완료: {summary_path}")

    return {'model': final_stacking_model, 'metrics': test_metrics}