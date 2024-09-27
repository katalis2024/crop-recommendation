 average values
    for param, pred_value in zip(['N', 'P', 'K', 'Temperature', 'Humidity', 'ph', 'Rainfall'],
                                 [N, P, K, Temperature, Humidity, ph, Rainfall]):
        avg_value = average_values[param]
        if pred_value < avg_value:
            recommendation.append(f"{param} deficient (prediction: {pred_value:.2f}), needs to be increased.")
        elif pred_value > avg_value:
            recommendation.append(f"{param} excess (prediction: {pred_value:.2f}), needs to be reduced.")
        else:
            recommendation.append(f"{param} adequate (prediction: {pred_value:.2f}).")

    return "\n".join(recommendation)


# Endpoint GET
@app.get("/")