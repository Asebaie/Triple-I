import joblib

loaded_bundle = joblib.load('model_bundle.pkl')

print(type(loaded_bundle))

print("Ключи в loaded_bundle:", loaded_bundle.keys())

print("\n--- Содержимое ---")

print(f"1. CatBoost модель ('cb'): {type(loaded_bundle['cb'])}")

print(f"\n2. LightGBM модель ('lgb'): {type(loaded_bundle['lgb'])}")

print(f"\n3. SHAP Explainer ('explainer'): {type(loaded_bundle['explainer'])}")

print(f"\n4. cat_features: {loaded_bundle['cat_features']}")

print(f"\n5. blend_weights: {loaded_bundle['blend_weights']}")

print("\n--- Параметры CatBoost модели ---")
print(loaded_bundle['cb'].get_params())

print("\n--- Краткая информация о LightGBM модели ---")
print(loaded_bundle['lgb'])

print(f"\n--- Тип SHAP Explainer ---")
print(type(loaded_bundle['explainer']))