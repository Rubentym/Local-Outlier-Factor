from sklearn.neighbors import LocalOutlierFactor

def detect_anomalies_lof(data):
    lof = LocalOutlierFactor(n_neighbors=2)
    outliers = lof.fit_predict(data)
    anomalies = data[outliers == -1]
    return anomalies

# Example usage:
data = [[1], [2], [3], [10], [15], [100]]
anomalies = detect_anomalies_lof(data)
print("Anomalies:", anomalies)
