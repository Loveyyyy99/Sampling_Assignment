from sklearn.metrics import accuracy_score
import pandas as pd

def evaluate_models(models, samplers, X, y):
    results = {}

    for s_name, sampler in samplers.items():
        X_s, y_s = sampler(X, y)
        results[s_name] = {}

        for m_name, model in models.items():
            model.fit(X_s, y_s)
            y_pred = model.predict(X_s)
            acc = accuracy_score(y_s, y_pred)
            results[s_name][m_name] = acc

    return pd.DataFrame(results)
