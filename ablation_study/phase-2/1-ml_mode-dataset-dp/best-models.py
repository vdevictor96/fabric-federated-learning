# Given data
data = {
    "twitter_dep": {
        "iid": {"fedavg": (81.50, 0.25), "fedprox": (81.82, 0.34)},
        "non_iid": {"fedavg": (74.98, 3.74), "fedprox": (74.27, 2.52)},
    },
    "acl_dep_sad": {
        "iid": {"fedavg": (91.74, 0.32), "fedprox": (90.51, 0.58)},
        "non_iid": {"fedavg": (77.90, 14.97), "fedprox": (76.10, 7.85)},
    },
    "mixed_depression": {
        "iid": {"fedavg": (88.12, 0.47), "fedprox": (87.23, 0.54)},
        "non_iid": {"fedavg": (72.05, 13.26), "fedprox": (84.04, 2.05)},
    },
    "dreaddit": {
        "iid": {"fedavg": (74.64, 0.63), "fedprox": (74.36, 0.57)},
        "non_iid": {"fedavg": (55.29, 0.56), "fedprox": (65.17, 1.70)},
    },
}

# Calculate "worst-case" performance and determine the best model
best_models = {dataset: {} for dataset in data}
for dataset, conditions in data.items():
    for condition, models in conditions.items():
        best_model = None
        best_performance = None
        for model, metrics in models.items():
            mean_accuracy, std_dev = metrics
            worst_case_performance = mean_accuracy - std_dev
            if best_performance is None or worst_case_performance > best_performance:
                best_performance = worst_case_performance
                best_model = model
        best_models[dataset][condition] = best_model

print(best_models)